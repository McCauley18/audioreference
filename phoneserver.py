"""
Real-time audio classification server.

Receives raw PCM16 audio chunks from the Expo app over a WebSocket,
maintains a rolling `max_seconds` window per connection, runs the
existing distilled CNN student model on that window at a fixed
cadence, and streams back JSON messages containing:

  - the predicted label / OOD decision
  - class probabilities
  - a compact mel-spectrogram frame for the current window (for
    display on the phone)

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Your phone and laptop must be on the same WiFi network. Find your
laptop's LAN IP (e.g. `ipconfig getifaddr en0` on macOS,
`ipconfig` on Windows) and point the Expo app at
ws://<that-ip>:8000/ws/stream
"""

import os
import json
import time
import asyncio
import logging

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio-server")

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "deployment_artifacts")

# How often (seconds) to run a fresh prediction on the rolling window.
PREDICT_INTERVAL_S = 1.0

# Number of mel bins sent to the client for the spectrogram display.
# Kept modest since this goes over the socket on every prediction tick.
N_MELS = 64
SPEC_TIME_FRAMES = 40  # downsampled time resolution sent to the client


# ---------------------------------------------------------------------------
# Model architecture (unchanged from the Streamlit version)
# ---------------------------------------------------------------------------
class AudioCNNStudent(nn.Module):
    def __init__(self, num_classes=3, groups=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
            nn.GroupNorm(min(groups, 32), 32), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(min(groups, 64), 64), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(min(groups, 128), 128), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(min(groups, 256), 256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        x = x.unsqueeze(1)
        feats = self.features(x)
        pooled = feats.squeeze(-1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits


# ---------------------------------------------------------------------------
# Inference pipeline: preprocessing must exactly match training
# ---------------------------------------------------------------------------
class InferencePipeline:
    def __init__(self, artifact_dir=ARTIFACT_DIR, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(artifact_dir, "config.json")) as f:
            self.config = json.load(f)

        self.class_names = self.config["class_names"]
        self.ood_label = self.config.get("ood_label", "OOD/Unknown")
        self.target_sr = self.config["target_sr"]
        self.max_seconds = self.config["max_seconds"]
        self.window_samples = self.target_sr * self.max_seconds

        self.model = AudioCNNStudent(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(artifact_dir, "student_model.pth"), map_location=self.device)
        )
        self.model.eval()

        # Mel-spectrogram transform, reused across predictions.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=self.window_samples // SPEC_TIME_FRAMES,
            n_mels=N_MELS,
            f_max=8000,
        ).to(self.device)

        logger.info(
            "Loaded model. classes=%s target_sr=%d max_seconds=%d device=%s",
            self.class_names, self.target_sr, self.max_seconds, self.device,
        )

    def _preprocess(self, window: np.ndarray) -> torch.Tensor:
        """window: float32 numpy array, already at target_sr, length == window_samples."""
        waveform = torch.from_numpy(window).float().to(self.device)
        mean, std = waveform.mean(), waveform.std()
        waveform = (waveform - mean) / std if std > 1e-8 else waveform - mean
        return waveform.unsqueeze(0)  # (1, T)

    @torch.no_grad()
    def predict(self, window: np.ndarray, energy_threshold=-1.0, conf_threshold=0.85):
        waveform = self._preprocess(window)
        logits, _ = self.model(waveform, return_features=True)

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        max_prob = float(probs[pred_idx])
        energy = float(-torch.logsumexp(logits, dim=1).item())

        is_ood = (energy > energy_threshold) and (max_prob < conf_threshold)
        label = self.ood_label if is_ood else self.class_names[pred_idx]

        return {
            "label": label,
            "class_probs": {n: float(p) for n, p in zip(self.class_names, probs)},
            "is_ood": is_ood,
            "energy": energy,
            "max_prob": max_prob,
        }

    @torch.no_grad()
    def spectrogram_frame(self, window: np.ndarray):
        """Compact mel-spectrogram (dB) for the current window, downsampled
        for cheap transport to the phone. Returns a list-of-lists (mels x time)."""
        waveform = torch.from_numpy(window).float().to(self.device).unsqueeze(0)
        mel = self.mel_transform(waveform)  # (1, n_mels, time)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
        mel_db = mel_db.squeeze(0).cpu().numpy()
        # Round for smaller JSON payloads.
        return np.round(mel_db, 1).tolist()


pipeline = InferencePipeline()

app = FastAPI()


class RollingBuffer:
    """Accumulates incoming PCM16 chunks into a float32 rolling window."""

    def __init__(self, window_samples: int):
        self.window_samples = window_samples
        self.buffer = np.zeros(window_samples, dtype=np.float32)
        self.filled = 0  # how many real samples we've received so far

    def push(self, pcm16_bytes: bytes): 
        chunk = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if chunk.size >= self.window_samples:
            self.buffer = chunk[-self.window_samples:]
            self.filled = self.window_samples
            return
        self.buffer = np.concatenate([self.buffer, chunk])[-self.window_samples:]
        self.filled = min(self.filled + chunk.size, self.window_samples)

    def is_ready(self) -> bool:
        # Require at least ~1s of real audio before predicting, so early
        # predictions aren't dominated by the zero-padding.
        return self.filled >= pipeline.target_sr


@app.websocket("/ws/stream")
async def stream_audio(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected: %s", ws.client)

    rolling = RollingBuffer(pipeline.window_samples)
    last_predict_time = 0.0

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message and message["bytes"] is not None:
                rolling.push(message["bytes"])

                now = time.monotonic()
                if rolling.is_ready() and (now - last_predict_time) >= PREDICT_INTERVAL_S:
                    last_predict_time = now
                    result = pipeline.predict(rolling.buffer)
                    result["spectrogram"] = pipeline.spectrogram_frame(rolling.buffer)
                    result["timestamp"] = now
                    await ws.send_text(json.dumps(result))

            elif "text" in message and message["text"] is not None:
            
                try:
                    control = json.loads(message["text"])
                    if control.get("type") == "reset":
                        rolling = RollingBuffer(pipeline.window_samples)
                        last_predict_time = 0.0
                        logger.info("Rolling buffer reset by client")
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", ws.client)


@app.get("/health")
async def health():
    return {"status": "ok", "classes": pipeline.class_names}