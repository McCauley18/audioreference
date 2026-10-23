import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import json
import tempfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Audio OOD Detection — Edge Student",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Audio OOD Detection — Distilled Edge Model")
st.markdown(
    "Upload a file **or** record from the microphone. "
    "Only audio above the silence gate is classified. "
    "OOD rejection uses **Energy score + confidence gate**."
)

ARTIFACT_DIR = "deployment_artifacts"

# -------------------- MODEL --------------------
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

    def forward(self, x, padding_mask=None, return_features=False):
        x = x.unsqueeze(1)
        feats = self.features(x)
        pooled = feats.squeeze(-1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits


class DeploymentPipeline:
    def __init__(self, artifact_dir=ARTIFACT_DIR, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(artifact_dir, "config.json")) as f:
            self.config = json.load(f)
        self.class_names = self.config["class_names"]
        self.target_sr = self.config["target_sr"]
        self.max_length = self.target_sr * self.config["max_seconds"]
        self.model = AudioCNNStudent(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(artifact_dir, "student_model.pth"), map_location=self.device)
        )
        self.model.eval()

    def _preprocess_waveform(self, waveform: torch.Tensor, sr: int):
        """waveform: (channels, samples) or (samples,)"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]
        elif waveform.shape[0] < self.max_length:
            waveform = F.pad(waveform, (0, self.max_length - waveform.shape[0]))
        mean, std = waveform.mean(), waveform.std()
        waveform = (waveform - mean) / std if std > 1e-8 else waveform - mean
        return waveform.unsqueeze(0).to(self.device)

    def _load_and_preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        return self._preprocess_waveform(waveform, sr)

    @torch.no_grad()
    def predict_from_waveform(
        self,
        waveform: torch.Tensor,
        sr: int,
        energy_threshold: float = -1.0,
        conf_threshold: float = 0.85,
    ):
        x = self._preprocess_waveform(waveform, sr)
        logits, _ = self.model(x, return_features=True)

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class_idx = int(probs.argmax())
        max_prob = float(probs[pred_class_idx])
        energy = float(-torch.logsumexp(logits, dim=1).item())

        is_ood = (energy > energy_threshold) and (max_prob < conf_threshold)
        label = self.config.get("ood_label", "OOD/Unknown") if is_ood else self.class_names[pred_class_idx]

        return {
            "label": label,
            "class_probs": {name: float(p) for name, p in zip(self.class_names, probs)},
            "is_ood": is_ood,
            "energy": energy,
            "max_prob": max_prob,
        }

    @torch.no_grad()
    def predict(self, audio_path, energy_threshold=-1.0, conf_threshold=0.85):
        waveform, sr = torchaudio.load(audio_path)
        return self.predict_from_waveform(waveform, sr, energy_threshold, conf_threshold)


@st.cache_resource
def load_pipeline():
    return DeploymentPipeline()


# -------------------- HELPERS --------------------
def compute_rms_db(waveform: np.ndarray):
    rms = np.sqrt(np.mean(waveform ** 2) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def plot_melspec(y, sr, title="Mel-spectrogram"):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, fmax=8000, cmap="viridis")
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close(fig)


# -------------------- MAIN UI --------------------
def main():
    pipeline = load_pipeline()
    class_labels = pipeline.class_names + [pipeline.config.get("ood_label", "OOD/Unknown")]

    # Sidebar
    with st.sidebar:
        st.header("Evaluation Settings")
        true_class = st.selectbox("True class:", class_labels, index=0)

        st.markdown("---")
        st.subheader("Gates")

        silence_threshold_db = st.slider(
            "Silence threshold (dB)",
            min_value=-60.0, max_value=-5.0, value=-28.0, step=1.0,
            help="Only classify if RMS energy is above this value."
        )

        energy_threshold = st.slider(
            "Energy threshold (OOD)",
            min_value=-5.0, max_value=5.0, value=-1.0, step=0.1,
        )

        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.50, max_value=0.99, value=0.85, step=0.01,
        )

        st.markdown("---")
        with st.expander("Model info"):
            st.write(f"Classes: {', '.join(pipeline.class_names)}")
            st.write(f"Sample rate: {pipeline.target_sr} Hz")
            st.write(f"Window: {pipeline.config['max_seconds']} s")

    # Main area
    tab1, tab2 = st.tabs(["Microphone (Live)", "File Upload"])

    # ========== TAB 1: MICROPHONE ==========
    with tab1:
        st.subheader("Record from microphone")
        st.caption("Click the microphone, speak / make a sound, then stop. The app will only classify if the clip is loud enough.")

        # Streamlit native audio input (recommended)
        audio_value = st.audio_input("Record a clip", key="mic_input")

        if audio_value is not None:
            # Save to temp file so we can use torchaudio / librosa easily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_value.getvalue())
                audio_path = tmp.name

            # Load for analysis
            y, sr = librosa.load(audio_path, sr=None)
            rms_db = compute_rms_db(y)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.audio(audio_value)
                st.write(f"**RMS level:** `{rms_db:.1f} dB`  |  Silence gate: `{silence_threshold_db:.0f} dB`")

                if rms_db < silence_threshold_db:
                    st.warning("Clip is too quiet (below silence gate). No classification performed.")
                    plot_melspec(y, sr, title="Mel-spectrogram (silent / ignored)")
                else:
                    st.success("Sound detected — running model…")
                    plot_melspec(y, sr)

                    result = pipeline.predict(
                        audio_path,
                        energy_threshold=energy_threshold,
                        conf_threshold=conf_threshold,
                    )

                    with col2:
                        st.subheader("Results")
                        st.metric("Predicted", result["label"])
                        st.metric("Confidence", f"{result['max_prob']*100:.1f}%")
                        st.metric("Energy", f"{result['energy']:.3f}")

                        if result["is_ood"]:
                            st.caption(
                                f"Flagged OOD — Energy ({result['energy']:.2f}) > {energy_threshold} "
                                f"**and** confidence ({result['max_prob']:.1%}) < {conf_threshold:.0%}."
                            )
                        else:
                            st.caption("Trusted CNN prediction.")

                        st.metric("True class", true_class)
                        if result["label"] == true_class:
                            st.success("Correct")
                        else:
                            st.error("Mismatch")

                        with st.expander("Class probabilities"):
                            for name, p in result["class_probs"].items():
                                st.write(f"{name}: {p*100:.2f}%")

                        with st.expander("Debug"):
                            st.json({
                                "energy_threshold": energy_threshold,
                                "conf_threshold": conf_threshold,
                                "energy": result["energy"],
                                "max_prob": result["max_prob"],
                                "is_ood": result["is_ood"],
                                "rms_db": float(rms_db),
                            })

            os.unlink(audio_path)

    # ========== TAB 2: FILE UPLOAD ==========
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=["wav", "flac", "mp3", "ogg"],
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getbuffer())
                audio_path = tmp.name

            y, sr = librosa.load(audio_path, sr=None)
            rms_db = compute_rms_db(y)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.audio(uploaded_file)
                st.write(f"**RMS level:** `{rms_db:.1f} dB`")
                plot_melspec(y, sr)

                if st.button("Evaluate file", type="primary"):
                    if rms_db < silence_threshold_db:
                        st.warning("File is below silence gate — treating as silent.")
                    else:
                        result = pipeline.predict(
                            audio_path,
                            energy_threshold=energy_threshold,
                            conf_threshold=conf_threshold,
                        )

                        with col2:
                            st.subheader("Results")
                            st.metric("Predicted", result["label"])
                            st.metric("Confidence", f"{result['max_prob']*100:.1f}%")
                            st.metric("Energy", f"{result['energy']:.3f}")

                            if result["is_ood"]:
                                st.caption("Flagged OOD")
                            else:
                                st.caption("Trusted CNN prediction")

                            st.metric("True class", true_class)
                            if result["label"] == true_class:
                                st.success("Correct")
                            else:
                                st.error("Mismatch")

                            with st.expander("Class probabilities"):
                                for name, p in result["class_probs"].items():
                                    st.write(f"{name}: {p*100:.2f}%")

            os.unlink(audio_path)


if __name__ == "__main__":
    main()