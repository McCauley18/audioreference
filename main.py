import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import json
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
import librosa
import librosa.display

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Audio Evaluation Tool (Edge Student Model)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Audio OOD Detection — Distilled Edge Model")
st.markdown(
    "Upload an audio file and get a prediction from the distilled CNN student "
    "model, with optional background/unknown-sound rejection via Local Outlier Factor."
)

ARTIFACT_DIR = "deployment_artifacts"


# STUDENT MODEL ARCHITECTURE
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


# DEPLOYMENT PIPELINE => model + scaler + LOF
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
        self.scaler = joblib.load(os.path.join(artifact_dir, "feature_scaler.pkl"))
        self.lof = joblib.load(os.path.join(artifact_dir, "lof_ood_detector.pkl"))

    def _load_and_preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
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
        return waveform.unsqueeze(0).to(self.device)  # (1, T)

    @torch.no_grad()
    def predict(self, audio_path, use_ood: bool = True, conf_threshold: float = 0.85):
        """
        Combined decision rule:
        - If use_ood=False → always trust CNN softmax.
        - If use_ood=True  → reject as OOD only when BOTH:
              1. LOF flags outlier
              2. max softmax probability < conf_threshold
          Otherwise return the CNN prediction.
        """
        waveform = self._load_and_preprocess(audio_path)
        padding_mask = torch.zeros(waveform.shape, dtype=torch.bool, device=self.device)
        logits, feats = self.model(waveform, padding_mask, return_features=True)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class_idx = int(probs.argmax())
        max_prob = float(probs[pred_class_idx])

        feats_scaled = self.scaler.transform(feats.cpu().numpy())
        is_outlier = self.lof.predict(feats_scaled)[0] == -1
        ood_score = float(self.lof.decision_function(feats_scaled)[0])

        # ----- Combined decision -----
        if use_ood and is_outlier and max_prob < conf_threshold:
            label = self.config["ood_label"]
            is_ood = True
        else:
            label = self.class_names[pred_class_idx]
            is_ood = False

        return {
            "label": label,
            "class_probs": {name: float(p) for name, p in zip(self.class_names, probs)},
            "is_ood": is_ood,
            "ood_score": ood_score,
            "max_prob": max_prob,
            "lof_outlier": bool(is_outlier),   # raw LOF decision (for debugging)
        }


@st.cache_resource
def load_pipeline():
    return DeploymentPipeline()


# MAIN STREAMLIT UI
def main():
    pipeline = load_pipeline()
    class_labels = pipeline.class_names + [pipeline.config["ood_label"]]

    with st.sidebar:
        st.header("Evaluation Settings")

        true_class = st.selectbox(
            "True class:",
            class_labels,
            index=0,
        )

        st.markdown("---")
        st.subheader("OOD / Rejection Controls")

        use_ood = st.toggle(
            "Enable LOF OOD detector",
            value=True,
            help="When disabled, the system always trusts the CNN softmax prediction."
        )

        conf_threshold = st.slider(
            "Confidence threshold (reject only if max-prob < this)",
            min_value=0.50,
            max_value=0.99,
            value=0.85,
            step=0.01,
            help="OOD is returned only when LOF says outlier AND max softmax probability is below this value."
        )

        st.markdown("---")
        st.info("Upload a file below to get a prediction from the edge-deployed student model.")

        with st.expander("Model info"):
            st.write(f"Classes: {', '.join(pipeline.class_names)}")
            st.write(f"Sample rate: {pipeline.target_sr} Hz")
            st.write(f"OOD method: Local Outlier Factor on student embeddings")
            st.write("Decision rule: LOF outlier **AND** max-prob < threshold → OOD, else CNN prediction")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=["wav", "flac", "mp3", "ogg"],
            help="Supported formats: WAV, FLAC, MP3, OGG.",
        )

        if uploaded_file is not None:
            st.markdown(f"**File:** `{uploaded_file.name}`")
            st.audio(uploaded_file, format="audio/wav")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                audio_path = tmp_file.name

            # Mel-spectrogram
            y, sr = librosa.load(audio_path, sr=pipeline.target_sr)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(8, 4))
            img = librosa.display.specshow(
                S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, fmax=8000, cmap="viridis"
            )
            ax.set_title("Mel-spectrogram")
            plt.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig)

            if st.button("Evaluate", type="primary"):
                result = pipeline.predict(
                    audio_path,
                    use_ood=use_ood,
                    conf_threshold=conf_threshold,
                )

                with col_right:
                    st.subheader("Results")
                    st.metric("Predicted Class", result["label"])

                    if result["is_ood"]:
                        st.metric("OOD Score", f"{result['ood_score']:.3f}")
                        st.caption(
                            "Flagged as unknown/background — LOF outlier **and** "
                            f"max-prob ({result['max_prob']:.1%}) < {conf_threshold:.0%}."
                        )
                    else:
                        st.metric("Confidence", f"{result['max_prob']*100:.1f}%")
                        st.metric("OOD Score", f"{result['ood_score']:.3f}")
                        if result["lof_outlier"]:
                            st.caption(
                                f"LOF wanted to reject (score={result['ood_score']:.3f}), "
                                f"but max-prob ({result['max_prob']:.1%}) ≥ {conf_threshold:.0%} "
                                "→ trusted CNN prediction."
                            )

                    st.metric("True Class (evaluation)", true_class)
                    if result["label"] == true_class:
                        st.success("Correct!")
                    else:
                        st.error("Mismatch")

                    with st.expander("Class probabilities"):
                        for name, p in result["class_probs"].items():
                            st.write(f"{name}: {p*100:.2f}%")

                    with st.expander("Debug info"):
                        st.write(f"LOF enabled: {use_ood}")
                        st.write(f"Confidence threshold: {conf_threshold:.2f}")
                        st.write(f"Raw LOF outlier: {result['lof_outlier']}")
                        st.write(f"Max softmax probability: {result['max_prob']:.4f}")
                        st.write(f"Final is_ood: {result['is_ood']}")

            os.unlink(audio_path)
        else:
            with col_right:
                st.info("Upload an audio file to begin.")


if __name__ == "__main__":
    main()