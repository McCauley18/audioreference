# app.py
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from BEATs import BEATs, BEATsConfig

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BEATS_CKPT = "BEATs_iter3_plus_AS2M.pt"
MODEL_CKPT = "beats_polyphonic_finetuned.pth"
CLASS_NAMES = ["Gunshot", "Human", "Vehicle"]
NUM_CLASSES = len(CLASS_NAMES)
TARGET_SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# MODEL
# -------------------------------------------------
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
# ADAPTIVE ENERGY GATE (lightweight pre-filter)
# -------------------------------------------------
class CFARGate:
    """
    Causal cell-averaging CFAR gate over fixed-size audio blocks.

    For each block n (the "cell under test"):
        threshold = alpha * mean(reference blocks)
        detect     = power[n] > threshold

    The reference blocks are the `n_ref` blocks immediately before the
    `n_guard` guard blocks that sit just before block n:

        ... [ R R R R R R ] [ G G ] [ CUT ]
              n_ref cells   guards    this block

    Guard cells stop a real event that straddles a block boundary from
    inflating its own noise estimate and masking itself.

    The comparison is done on LINEAR block power (mean of x^2), because the
    "constant false alarm rate" guarantee is derived for power on a linear
    scale. dB is only used for plotting.

    This is the offline (whole-clip) form: it uses a plain slice + mean for
    readability. On a phone you'd keep a ring buffer with a running sum so
    each block costs O(1) (see the C version).
    """

    def __init__(
        self,
        sample_rate=TARGET_SR,
        block_ms=10,
        n_ref=16,
        n_guard=2,
        alpha=8.0,
        min_active_ms=20,
        hangover_ms=100,
    ):
        self.sr = sample_rate
        self.block = max(1, int(round(sample_rate * block_ms / 1000.0)))
        self.block_ms = block_ms
        self.n_ref = max(1, int(n_ref))
        self.n_guard = max(0, int(n_guard))
        self.alpha = float(alpha)

        self.L = self.n_ref + self.n_guard          # history needed
        self.min_active = max(1, int(round(min_active_ms / block_ms)))
        self.hangover = max(0, int(round(hangover_ms / block_ms)))

    # ---------- theory helpers ----------
    @staticmethod
    def pfa(n_ref, alpha):
        """Expected false-alarm rate per block on stationary Gaussian noise."""
        return (1.0 + alpha / float(n_ref)) ** (-int(n_ref))

    @staticmethod
    def alpha_from_pfa(n_ref, pfa):
        """Inverse of the above: pick alpha for a target false-alarm rate."""
        return float(n_ref) * (float(pfa) ** (-1.0 / int(n_ref)) - 1.0)

    # ---------- main entry point ----------
    def process(self, wav: np.ndarray):
        """
        Returns:
          times         : centre time of each block (s)
          power_db      : block loudness (dB, 10*log10 of mean square)
          floors_db     : estimated noise floor per block (dB). NaN during warm-up.
          flags         : bool mask - True = block is worth sending to the model
        """
        wav = np.asarray(wav, dtype=np.float64)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)

        # --- chop into non-overlapping blocks, pad the tail ---
        n_blocks = max(1, int(np.ceil(len(wav) / self.block)))
        pad = n_blocks * self.block - len(wav)
        if pad:
            wav = np.pad(wav, (0, pad))

        blocks = wav.reshape(n_blocks, self.block)

        # --- block power (linear!) and its dB twin for plotting ---
        power = np.mean(blocks ** 2, axis=1)
        power_db = 10.0 * np.log10(power + 1e-12)
        times = (np.arange(n_blocks) + 0.5) * self.block / self.sr

        floors_db = np.full(n_blocks, np.nan, dtype=np.float64)
        detect = np.zeros(n_blocks, dtype=bool)

        # --- the CFAR loop ---
        for n in range(self.L, n_blocks):
            ref = power[n - self.L: n - self.n_guard]   # n_ref cells
            ref_mean = ref.mean()
            floors_db[n] = 10.0 * np.log10(ref_mean + 1e-12)
            detect[n] = power[n] > self.alpha * ref_mean

        # --- debounce + hangover, so single-block blips don't trip the gate ---
        flags = np.zeros(n_blocks, dtype=bool)
        run = 0
        hold = 0
        for n in range(n_blocks):
            if detect[n]:
                run += 1
                hold = self.hangover
            else:
                run = 0
                hold = max(0, hold - 1)
            flags[n] = (run >= self.min_active) or (hold > 0)

        return times, power_db, floors_db, flags


def window_is_interesting(gate_times, gate_flags, start_s, end_s, min_ratio=0.15):
    """True if enough gated frames fall inside [start_s, end_s]."""
    if len(gate_times) == 0:
        return True
    in_win = (gate_times >= start_s) & (gate_times <= end_s)
    if not np.any(in_win):
        return False
    return float(gate_flags[in_win].mean()) >= min_ratio

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def load_audio(path_or_bytes, target_sr=TARGET_SR):
    if isinstance(path_or_bytes, (str, Path)):
        wav, _ = librosa.load(path_or_bytes, sr=target_sr, mono=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(path_or_bytes)
            tmp = f.name
        wav, _ = librosa.load(tmp, sr=target_sr, mono=True)
        os.unlink(tmp)
    return torch.from_numpy(wav).float()

def estimate_snr(wav, frame_len=400, hop=160):
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
    freqs = np.fft.rfftfreq(len(wav), 1 / sr)
    mag = np.abs(np.fft.rfft(wav))
    return freqs[np.argmax(mag)]

def sliding_windows(waveform, win_s=2.5, hop_s=2.5):
    win = int(win_s * TARGET_SR)
    hop = int(hop_s * TARGET_SR)
    starts = list(range(0, max(1, waveform.numel() - win + 1), hop)) or [0]
    return starts, win

# -------------------------------------------------
# UI
# -------------------------------------------------
st.set_page_config(
    page_title="Audio Event Detector (BEATs + Adaptive Gate)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Gunshot | Human | Vehicle Detector")
st.markdown(
    "Upload or record a clip."
)
# A **lightweight adaptive energy gate** first finds "
#     "moments that stand out from the local background; **BEATs** only runs on those "
#     "windows (same idea as the phone-side battery-saving gate)."
# ---- Sidebar ----
with st.sidebar:
    st.header("Evaluation Settings")

    win_seconds = st.slider("Window length (seconds)", 1.5, 4.0, 2.5, 0.5)
    hop_seconds = st.slider("Hop length (seconds)", 1.0, 4.0, 2.5, 0.5)

    st.markdown("---")
    # st.subheader("CFAR gate (pre-filter)")
    # st.caption(
    #     "Cell-averaging CFAR: each block is compared to the average of the "
    #     "recent blocks before it, so the threshold follows the noise. "
    #     "Keeps the false-alarm rate roughly constant."
    # )
    use_gate = st.checkbox("Enable CFAR gate", value=True)

    block_ms = st.slider("Block size (ms)", 5, 40, 10, 5)
    n_ref = st.slider("Reference cells (n_ref)", 4, 64, 16, 2)
    n_guard = st.slider("Guard cells (n_guard)", 1, 8, 2, 1)
    alpha = st.slider("Threshold multiplier (alpha)", 1.0, 20.0, 8.0, 0.5)

    _pfa = CFARGate.pfa(n_ref, alpha)
    st.caption(f"≈ {_pfa:.1e} false alarms per block on stationary noise "
               f"({_pfa * 1000.0 / block_ms:.2f} / s)")

    min_active_ms = st.slider("Min active duration (ms)", 0, 200, 20, 10)
    hangover_ms = st.slider("Hangover after detection (ms)", 0, 500, 100, 50)
    min_gate_ratio = st.slider(
        "Min fraction of window marked interesting", 0.05, 0.50, 0.15, 0.05
    )

    st.markdown("---")
    st.subheader("OOD / Quality Thresholds")
    max_prob_thresh = st.slider("Min max-sigmoid probability", 0.10, 0.95, 0.45, 0.05)
    snr_thresh = st.slider("Min SNR (dB)", 0.0, 20.0, 6.0, 0.5)
    freq_min = st.number_input("Min dominant freq (Hz)", 50, 1000, 80)
    freq_max = st.number_input("Max dominant freq (Hz)", 2000, 8000, 7500)

# ---- Input ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Record")
    audio_bytes = st.audio_input("Click to record")

with col2:
    st.subheader("Or upload a file")
    uploaded = st.file_uploader("WAV | FLAC | MP3 | AAC", type=["wav", "flac", "mp3", "aac"])

# ---- Process ----
if audio_bytes is not None or uploaded is not None:
    if audio_bytes is not None:
        waveform = load_audio(audio_bytes.getvalue())
        source = "Live recording"
        play_bytes = audio_bytes
    else:
        waveform = load_audio(uploaded.getvalue())
        source = uploaded.name
        play_bytes = uploaded

    duration = waveform.numel() / TARGET_SR
    st.success(f"Loaded **{source}**  •  {duration:.1f}s  •  {TARGET_SR} Hz")
    st.audio(play_bytes)

    # Full-clip spectrogram
    st.subheader("Mel Spectrogram")
    y = waveform.numpy()
    S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    img = librosa.display.specshow(
        S_dB, sr=TARGET_SR, x_axis="time", y_axis="mel", ax=ax, fmax=8000, cmap="viridis"
    )
    ax.set_title("Mel-spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close(fig)

    if st.button("Evaluate", type="primary"):
        with st.spinner("Running adaptive gate + sliding-window inference..."):
            # ----- 1) Adaptive gate over the whole clip -----
            gate = CFARGate(
                sample_rate=TARGET_SR,
                block_ms=block_ms,
                n_ref=n_ref,
                n_guard=n_guard,
                alpha=alpha,
                min_active_ms=min_active_ms,
                hangover_ms=hangover_ms,
            ) 
            g_times, g_energy, g_floor, g_flags = gate.process(y)

            # Plot gate vs energy
            st.subheader("Audio keeper")
            fig_g, ax_g = plt.subplots(figsize=(10, 3))
            ax_g.plot(g_times, g_energy, label="Frame loudness (dB)", linewidth=1.0)
            ax_g.plot(g_times, g_floor, label="Adaptive background (noise floor)", linewidth=1.2)
            alpha_db = 10.0 * np.log10(alpha)   # alpha in dB == CFAR threshold offset
            ax_g.plot( 
                g_times,
                g_floor + alpha_db,
                "--",
                label=f"CFAR threshold (alpha={alpha:.1f} ≈ +{alpha_db:.1f} dB)",
                linewidth=1.0,
            )
            if np.any(g_flags):
                ax_g.fill_between(
                    g_times,
                    g_energy.min() - 5,
                    g_energy.max() + 5,
                    where=g_flags,
                    alpha=0.25,
                    label="Interesting (gate open)",
                )
            ax_g.set_xlabel("Time (s)")
            ax_g.set_ylabel("dB")
            ax_g.legend(loc="upper right", fontsize=8)
            ax_g.set_title("Adaptive energy gate")
            st.pyplot(fig_g)
            plt.close(fig_g)

            # ----- 2) Sliding windows; BEATs only if gate says interesting -----
            starts, win = sliding_windows(waveform, win_s=win_seconds, hop_s=hop_seconds)

            all_logits = []
            all_starts = []
            decisions = []
            max_probs = []
            snrs = []
            freqs = []
            gate_status = []
            model_ran = []

            with torch.no_grad():
                for start in starts:
                    start_s = start / TARGET_SR
                    end_s = start_s + win_seconds
                    chunk = waveform[start : start + win]
                    if chunk.numel() < win:
                        chunk = F.pad(chunk, (0, win - chunk.numel()))

                    interesting = (
                        True
                        if not use_gate
                        else window_is_interesting(
                            g_times, g_flags, start_s, end_s, min_ratio=min_gate_ratio
                        )
                    )

                    snr = estimate_snr(chunk)
                    dom_freq = estimate_dominant_freq(chunk)

                    if not interesting:
                        # Skip BEATs — treat as background for this window
                        decision = "Background"
                        max_p = 0.0
                        logits = np.zeros(NUM_CLASSES, dtype=np.float32)
                        ran = False
                        gstat = "skipped"
                    else:
                        chunk_t = chunk.unsqueeze(0).to(DEVICE)
                        mask = torch.zeros(1, win, dtype=torch.bool, device=DEVICE)
                        logits = model(chunk_t, mask).cpu().numpy()[0]
                        probs = 1 / (1 + np.exp(-logits))
                        max_p = float(probs.max())
                        ran = True
                        gstat = "open"

                        quality_ok = (snr >= snr_thresh) and (freq_min <= dom_freq <= freq_max)

                        if max_p < max_prob_thresh:
                            decision = "Background"
                        else:
                            active = [
                                CLASS_NAMES[j]
                                for j in range(NUM_CLASSES)
                                if probs[j] > 0.5
                            ]
                            base = " + ".join(active) if active else "Unknown (low conf)"
                            decision = base if quality_ok else f"{base}"

                    all_logits.append(logits)
                    all_starts.append(start_s)
                    decisions.append(decision)
                    max_probs.append(max_p)
                    snrs.append(snr)
                    freqs.append(dom_freq)
                    gate_status.append(gstat)
                    model_ran.append(ran)

            n_total = len(starts)
            n_ran = sum(model_ran)
            st.info(
                f"Adaptive gate: **{n_ran}/{n_total}** windows sent to BEATs "
                f"({100.0 * n_ran / max(1, n_total):.0f}% model use)."
            )

            # Results table
            st.subheader("Window-level results")
            df = pd.DataFrame({
                "Start (s)": [f"{s:.1f}" for s in all_starts],
                "End (s)": [f"{s + win_seconds:.1f}" for s in all_starts],
                "Gate": gate_status,
                "Decision": decisions,
                "Max Prob": [f"{p:.2f}" for p in max_probs],
                "SNR (dB)": [f"{s:.1f}" for s in snrs],
                "Dom Freq (Hz)": [f"{f:.0f}" for f in freqs],
            })
            st.dataframe(df, use_container_width=True)

            # Majority over non-gated-background decisions when possible
            # vote_pool = [d for d in decisions if d != "Background (gated)"]
            # if not vote_pool:
            #     vote_pool = decisions
            # final = max(set(vote_pool), key=vote_pool.count)
            # st.markdown(f"### Final clip decision (majority): **`{final}`**")

            # Probability bars (only windows where model ran)
            # st.subheader("Class probabilities (windows where model ran)")
            # ran_idx = [i for i, r in enumerate(model_ran) if r]
            # if ran_idx:
            #     probs_arr = 1 / (1 + np.exp(-np.array([all_logits[i] for i in ran_idx])))
            #     n_show = min(8, len(probs_arr))
            #     fig2, ax2 = plt.subplots(figsize=(10, 4))
            #     x = np.arange(n_show)
            #     width = 0.25
            #     for c, name in enumerate(CLASS_NAMES):
            #         ax2.bar(x + c * width, probs_arr[:n_show, c], width, label=name)
            #     ax2.set_xticks(x + width)
            #     ax2.set_xticklabels([f"{all_starts[ran_idx[i]]:.1f}s" for i in range(n_show)])
            #     ax2.set_ylim(0, 1.05)
            #     ax2.legend()
            #     ax2.set_ylabel("Sigmoid probability")
            #     st.pyplot(fig2)
            #     plt.close(fig2) 
            # else:
            #     st.warning("No windows passed the adaptive gate — try lowering the margin (dB).")
      
else:
    st.info("Record or upload an audio file to start.")