# live_detector.py
"""
Real-time audio event detector on laptop.

Pipeline:
    mic → rolling buffer → streaming CFAR → (on rising edge) → BEATs → print/save

Usage:
    python live_detector.py                    # defaults
    python live_detector.py --list-devices     # show microphones
    python live_detector.py --device 2         # pick a mic
    python live_detector.py --save-clips       # save every triggered clip to ./detections/

Ctrl+C to stop.
"""
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import argparse
import queue
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import sounddevice as sd
from scipy.io import wavfile

from BEATs import BEATs, BEATsConfig


# ============================================================
# CONFIG  
# ============================================================
BEATS_CKPT  = "BEATs_iter3_plus_AS2M.pt"
MODEL_CKPT  = "beats_polyphonic_finetuned.pth"
CLASS_NAMES = ["Gunshot", "Human", "Vehicle"]
NUM_CLASSES = len(CLASS_NAMES)
TARGET_SR   = 16000
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIN_S       = 2.5
WIN_SAMPLES = int(TARGET_SR * WIN_S)

# CFAR defaults (mirror the Streamlit app)
BLOCK_MS      = 10
N_REF         = 16
N_GUARD       = 2
ALPHA         = 8.0
MIN_ACTIVE_MS = 20
HANGOVER_MS   = 100

# Gate → model policy
COOLDOWN_S  = 0.5     # min seconds between model calls
PROB_THRESH = 0.5     # label a class if sigmoid > this


# ============================================================
# MODEL
# ============================================================
class BEATsClassifier(nn.Module):
    def __init__(self, beats_model, num_classes):
        super().__init__()
        self.beats = beats_model
        self.classifier = nn.Linear(beats_model.cfg.encoder_embed_dim, num_classes)

    def forward(self, x, padding_mask=None):
        features = self.beats.extract_features(x, padding_mask)[0]
        pooled = features.mean(dim=1)
        return self.classifier(pooled)


def load_model():
    print(f"[model] loading on {DEVICE}…", flush=True)
    checkpoint = torch.load(BEATS_CKPT, map_location="cpu")
    config = BEATsConfig(checkpoint["cfg"])
    audio_model = BEATs(config)
    audio_model.load_state_dict(checkpoint["model"])

    model = BEATsClassifier(audio_model, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location="cpu"))
    model.to(DEVICE).eval()
    print("[model] ready", flush=True)
    return model


# ============================================================
# STREAMING CFAR GATE
# ============================================================
class StreamingCFARGate:
    """
    Stateful cell-averaging CFAR. Push audio in, get one record per
    completed block. Numerically identical to CFARGate.process() on
    the same audio — just incremental.

    Ring buffer invariant once warm:
        power_buf[head : head+n_ref]  (mod L) == the n_ref reference cells
        for the *next* cell under test.
    """
    def __init__(self, sr=TARGET_SR, block_ms=BLOCK_MS, n_ref=N_REF,
                 n_guard=N_GUARD, alpha=ALPHA, min_active_ms=MIN_ACTIVE_MS,
                 hangover_ms=HANGOVER_MS):
        self.sr = sr
        self.block_size = max(1, int(sr * block_ms / 1000.0))
        self.block_ms = block_ms
        self.n_ref = max(1, int(n_ref))
        self.n_guard = max(0, int(n_guard))
        self.alpha = float(alpha)
        self.L = self.n_ref + self.n_guard
        self.min_active = max(1, int(round(min_active_ms / block_ms)))
        self.hangover = max(0, int(round(hangover_ms / block_ms)))

        self.power_buf = np.zeros(self.L, dtype=np.float64)
        self.ref_sum = 0.0
        self.head = 0
        self.count = 0

        self.run = 0
        self.hold = 0

        self.sample_buf = np.zeros(0, dtype=np.float32)
        self.block_idx = 0

    def push(self, samples):
        if len(samples) == 0:
            return []
        self.sample_buf = np.concatenate(
            [self.sample_buf, samples.astype(np.float32, copy=False)]
        )
        out = []
        while len(self.sample_buf) >= self.block_size:
            chunk = self.sample_buf[: self.block_size]
            self.sample_buf = self.sample_buf[self.block_size:]
            power = float(np.mean(chunk.astype(np.float64) ** 2))

            detect = False
            floor_db = float("nan")

            if self.count == self.L:
                ref_mean = self.ref_sum / self.n_ref
                floor_db = 10.0 * np.log10(ref_mean + 1e-12)
                detect = power > self.alpha * ref_mean
                leaving  = self.power_buf[self.head]
                entering = self.power_buf[(self.head + self.n_ref) % self.L]
                self.ref_sum = self.ref_sum - leaving + entering

            self.power_buf[self.head] = power
            self.head = (self.head + 1) % self.L
            if self.count < self.L:
                self.count += 1
                if self.count == self.L:
                    self.ref_sum = sum(
                        self.power_buf[(self.head + i) % self.L]
                        for i in range(self.n_ref)
                    )

            if detect:
                self.run += 1
                self.hold = self.hangover
            else:
                self.run = 0
                self.hold = max(0, self.hold - 1)
            flag = (self.run >= self.min_active) or (self.hold > 0)

            out.append({
                "t":        self.block_idx * self.block_size / self.sr,
                "power_db": 10.0 * np.log10(power + 1e-12),
                "floor_db": floor_db,
                "detect":   detect,
                "flag":     flag,
            })
            self.block_idx += 1
        return out


# ============================================================
# ROLLING BUFFER (holds last WIN_S seconds)
# ============================================================
class RollingBuffer:
    def __init__(self, length):
        self.length = int(length)
        self.buf = np.zeros(self.length, dtype=np.float32)

    def push(self, samples):
        n = len(samples)
        if n == 0:
            return
        if n >= self.length:
            self.buf[:] = samples[-self.length:]
        else:
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = samples

    def get(self):
        return self.buf.copy()


# ============================================================
# INFERENCE WORKER (runs BEATs off the audio path)
# ============================================================
class InferenceWorker:
    def __init__(self, model, device, class_names, save_dir=None):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.save_dir = save_dir
        self.lock = threading.Lock()

    def busy(self):
        return self.lock.locked()

    def submit(self, audio, trigger_t):
        if self.busy():
            return False
        t = threading.Thread(
            target=self._run, args=(audio, trigger_t), daemon=True
        )
        t.start()
        return True

    def _run(self, audio, trigger_t):
        with self.lock:
            try:
                t0 = time.time()
                x = torch.from_numpy(audio).unsqueeze(0).to(self.device)
                mask = torch.zeros(1, len(audio), dtype=torch.bool,
                                   device=self.device)
                with torch.no_grad():
                    logits = self.model(x, mask).cpu().numpy()[0]
                probs = 1.0 / (1.0 + np.exp(-logits))
                dt_ms = (time.time() - t0) * 1000.0

                labels = [
                    (self.class_names[j], float(probs[j]))
                    for j in range(len(self.class_names))
                    if probs[j] > PROB_THRESH
                ]
                if labels:
                    text = " + ".join(f"{n} ({p:.2f})" for n, p in labels)
                else:
                    top = int(np.argmax(probs))
                    text = (f"(none > {PROB_THRESH:.2f}; "
                            f"max {self.class_names[top]} {probs[top]:.2f})")

                print(f"[{trigger_t:6.2f}s] 🎯 {text}   [{dt_ms:.0f} ms]",
                      flush=True)

                if self.save_dir is not None:
                    self._save(audio, trigger_t)
            except Exception as e:
                print(f"[inference error] {e}", file=sys.stderr, flush=True)

    def _save(self, audio, t):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fname = self.save_dir / (
            f"event_{time.strftime('%Y%m%d_%H%M%S')}_{int(t*1000):05d}.wav"
        )
        wavfile.write(str(fname), TARGET_SR,
                      (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
        print(f"          saved → {fname}", flush=True)


# ============================================================
# CLI / SETUP
# ============================================================
def list_audio_devices():
    print(sd.query_devices())
    print()
    try:
        default = sd.default.device
        print(f"Default input index: {default[0]}")
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Real-time audio event detector")
    p.add_argument("--device", type=int, default=None,
                   help="Input device index (see --list-devices)")
    p.add_argument("--save-clips", action="store_true",
                   help="Save each triggered clip to ./detections/")
    p.add_argument("--list-devices", action="store_true",
                   help="List audio devices and exit")
    p.add_argument("--cooldown", type=float, default=COOLDOWN_S,
                   help=f"Min seconds between model calls (default: {COOLDOWN_S})")
    return p.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    for f in (BEATS_CKPT, MODEL_CKPT):
        if not Path(f).exists():
            print(f"[error] missing: {f}", file=sys.stderr)
            sys.exit(1)

    model = load_model()
    gate = StreamingCFARGate()
    rolling = RollingBuffer(WIN_SAMPLES)

    save_dir = Path("detections") if args.save_clips else None
    worker = InferenceWorker(model, DEVICE, CLASS_NAMES, save_dir=save_dir)

    audio_q = queue.Queue(maxsize=200)   # ~2 s of blocks

    def cb(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr, flush=True)
        chunk = indata.mean(axis=1).astype(np.float32, copy=False)
        try:
            audio_q.put_nowait(chunk)
        except queue.Full:
            pass   # drop rather than block the audio thread

    print("=" * 62)
    print("  Real-time Audio Event Detector")
    print(f"  Model   : BEATs ({', '.join(CLASS_NAMES)})")
    print(f"  Device  : {DEVICE}")
    print(f"  Window  : {WIN_S:.1f}s @ {TARGET_SR} Hz")
    print(f"  CFAR    : block={BLOCK_MS}ms  n_ref={N_REF}  "
          f"n_guard={N_GUARD}  alpha={ALPHA}")
    print(f"  Cooldown: {args.cooldown:.2f}s")
    if save_dir:
        print(f"  Saving  : {save_dir.resolve()}/")
    print("=" * 62)
    print("\nListening…  (Ctrl+C to stop)\n", flush=True)

    was_open = False
    last_infer_t = -1e9

    try:
        with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                            blocksize=int(TARGET_SR * BLOCK_MS / 1000.0),
                            device=args.device, callback=cb):
            while True:
                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                rolling.push(chunk)
                for r in gate.push(chunk):
                    rising = r["flag"] and not was_open
                    was_open = r["flag"]

                    if not rising:
                        continue
                    if (r["t"] - last_infer_t) < args.cooldown:
                        continue
                    if worker.busy():
                        continue

                    last_infer_t = r["t"]
                    snapshot = rolling.get()
                    worker.submit(snapshot, r["t"])

    except sd.PortAudioError as e:
        print(f"\n[error] cannot open microphone at {TARGET_SR} Hz: {e}",
              file=sys.stderr)
        print("Try: python live_detector.py --list-devices", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()