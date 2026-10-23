# app.py
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"


import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import tempfile
import os
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
  
# -------------------------------------------------
# CONFIG – edit paths if needed
# -------------------------------------------------
BEATS_CKPT = "BEATs_iter3_plus_AS2M.pt"          # original BEATs weights
MODEL_CKPT = "beats_polyphonic_finetuned.pth"    # your fine-tuned head
CLASS_NAMES = ["Gunshot", "Human", "Vehicle"]
NUM_CLASSES = len(CLASS_NAMES)
TARGET_SR = 16000
WIN_SECONDS = 1.0
HOP_SECONDS = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# BEATs + Classifier (same as training)
# -------------------------------------------------
# You need the official BEATs classes in the same folder
from BEATs import BEATs, BEATsConfig   # make sure BEATs.py is available

class BEATsClassifier(nn.Module):
    def __init__(self, beats_model, num_classes):
        super().__init__()
        self.beats = beats_model
        self.classifier = nn.Linear(beats_model.cfg.encoder_embed_dim, num_classes)

    def forward(self, x, padding_mask=None, return_features=False):
        features = self.beats.extract_features(x, padding_mask)[0]
        pooled = features.mean(dim=1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits

@st.cache_resource
def load_model():
    checkpoint = torch.load(BEATS_CKPT, map_location="cpu")
    config = BEATsConfig(checkpoint["cfg"])
    audio_model = BEATs(config)
    audio_model.load_state_dict(checkpoint["model"])

    model = BEATsClassifier(audio_model, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location="cpu"))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def load_wav(path_or_bytes, target_sr=TARGET_SR):
    # If it's a file path (string or Path)
    if isinstance(path_or_bytes, (str, Path)):
        wav, sr = librosa.load(path_or_bytes, sr=target_sr, mono=True)
    else:
        # If it's bytes from an uploaded file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(path_or_bytes)
            tmp = f.name
        wav, sr = librosa.load(tmp, sr=target_sr, mono=True)
        os.unlink(tmp)
    # librosa returns float32 in [-1, 1], convert to torch tensor
    return torch.from_numpy(wav).float()

def estimate_snr(wav, frame_len=400, hop=160):
    """Very simple energy-based SNR estimate (dB)."""
    wav = wav.numpy() if torch.is_tensor(wav) else wav
    energy = librosa.feature.rms(y=wav, frame_length=frame_len, hop_length=hop)[0]
    if len(energy) < 2:
        return 0.0
    noise_floor = np.percentile(energy, 10)
    signal = np.percentile(energy, 90)
    return 20 * np.log10((signal + 1e-8) / (noise_floor + 1e-8))

def estimate_dominant_freq(wav, sr=TARGET_SR):
    wav = wav.numpy() if torch.is_tensor(wav) else wav
    if len(wav) < 512:
        return 0.0
    freqs = np.fft.rfftfreq(len(wav), 1/sr)
    mag = np.abs(np.fft.rfft(wav))
    return freqs[np.argmax(mag)]

def sliding_inference(model, waveform, win_s=WIN_SECONDS, hop_s=HOP_SECONDS):
    win = int(win_s * TARGET_SR)
    hop = int(hop_s * TARGET_SR)
    starts = list(range(0, max(1, waveform.numel() - win + 1), hop)) or [0]

    all_logits, all_embs, all_starts = [], [], []

    with torch.no_grad():
        for start in starts:
            chunk = waveform[start:start + win]
            if chunk.numel() < win:
                chunk = F.pad(chunk, (0, win - chunk.numel()))
            chunk = chunk.unsqueeze(0).to(DEVICE)
            mask = torch.zeros(1, win, dtype=torch.bool, device=DEVICE)
            logits, emb = model(chunk, mask, return_features=True)
            all_logits.append(logits.cpu().numpy()[0])
            all_embs.append(emb.cpu().numpy()[0])
            all_starts.append(start / TARGET_SR)

    return np.array(all_logits), np.array(all_embs), all_starts

def energy_score(logits):
    """Higher = more ID-like (standard energy OOD score is actually -energy, we flip it)."""
    return -np.log(np.sum(np.exp(logits), axis=-1) + 1e-8)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Audio Event Detector + OOD", layout="wide")
st.title("Gunshot / Human / Vehicle Detector  +  OOD Rejection")

st.markdown("""
Record or upload a short audio clip.  
The model runs **sliding 1-second windows** and decides for each window:
- One (or more) of the three known classes, **or**
- **Unknown / Background** if the OOD score or quality gates fail.
""")

# ---- Sidebar: thresholds ----
st.sidebar.header("Decision Thresholds")
max_prob_thresh = st.sidebar.slider("Min max-sigmoid probability", 0.1, 0.95, 0.45, 0.05)
energy_thresh   = st.sidebar.slider("Min energy score (higher = more ID)", -10.0, 5.0, -1.5, 0.1)
snr_thresh      = st.sidebar.slider("Min SNR (dB)", 0.0, 20.0, 6.0, 0.5)
freq_min        = st.sidebar.number_input("Min dominant freq (Hz)", 50, 1000, 80)
freq_max        = st.sidebar.number_input("Max dominant freq (Hz)", 2000, 8000, 7500)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Start with the default values, then lower the probability threshold if the model is too conservative.")

# ---- Input ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Record live")
    audio_bytes = st.audio_input("Click to record (3–10 seconds recommended)")

with col2: 
    st.subheader("Or upload a file")
    uploaded = st.file_uploader("WAV / FLAC / MP3", type=["wav", "flac", "mp3"])

# ---- Process ----
if audio_bytes is not None or uploaded is not None:
    with st.spinner("Running inference..."):
        if audio_bytes is not None:
            waveform = load_wav(audio_bytes.getvalue())
            source = "Live recording"
        else:
            waveform = load_wav(uploaded.getvalue())
            source = uploaded.name

        duration = waveform.numel() / TARGET_SR
        st.success(f"Loaded **{source}**  •  {duration:.1f}s  •  {TARGET_SR} Hz")

        # Quality gates on the whole clip
        snr = estimate_snr(waveform)
        dom_freq = estimate_dominant_freq(waveform)

        st.write(f"**Estimated SNR:** {snr:.1f} dB   |   **Dominant frequency:** {dom_freq:.0f} Hz")

        # Sliding windows
        logits, embs, starts = sliding_inference(model, waveform)
        probs = 1 / (1 + np.exp(-logits))          # sigmoid
        max_probs = probs.max(axis=1)
        energies = energy_score(logits)

        # Decision per window
        decisions = []
        for i in range(len(starts)):
            # Quality gate
            if snr < snr_thresh or not (freq_min <= dom_freq <= freq_max):
                decisions.append("Unknown (quality)")
                continue

            # OOD check
            if max_probs[i] < max_prob_thresh or energies[i] < energy_thresh:
                decisions.append("Unknown / OOD")
            else:
                # Multi-label: all classes above 0.5
                active = [CLASS_NAMES[j] for j in range(NUM_CLASSES) if probs[i, j] > 0.5]
                decisions.append(" + ".join(active) if active else "Unknown (low conf)")

        # ---- Results table ----
        st.subheader("Window-level results")
        import pandas as pd
        df = pd.DataFrame({
            "Start (s)": [f"{s:.1f}" for s in starts],
            "Max Prob": [f"{p:.2f}" for p in max_probs],
            "Energy": [f"{e:.2f}" for e in energies],
            "Decision": decisions
        })
        st.dataframe(df, use_container_width=True)

        # ---- Summary ----
        final = max(set(decisions), key=decisions.count)   # most common decision
        st.markdown(f"### Final clip decision: **`{final}`**")

        # ---- Probability bars for first few windows ----
        st.subheader("Class probabilities (first 8 windows)")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(min(8, len(probs)))
        width = 0.25
        for c, name in enumerate(CLASS_NAMES):
            ax.bar(x + c*width, probs[:8, c], width, label=name)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{s:.1f}s" for s in starts[:8]])
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_ylabel("Sigmoid probability")
        st.pyplot(fig)

else:
    st.info("Record or upload an audio file to start.")