"""
Export the distilled AudioCNNStudent model to ONNX so it can run fully
client-side in the browser with onnxruntime-web (no server needed).

USAGE
-----
Run this in the same environment where you trained / saved the model,
right next to your existing "deployment_artifacts" folder:

    pip install onnx
    python export_to_onnx.py

INPUT  (expected in ./deployment_artifacts/):
    - config.json         (class_names, target_sr, max_seconds, ood_label)
    - student_model.pth   (state_dict for AudioCNNStudent)

OUTPUT (written to ./web_export/):
    - student_model.onnx
    - config.json          (copied over unchanged)

Upload BOTH files from web_export/ into the mobile web app's
"Load model" section (the .onnx file and the .json file).
"""
import os
import json
import shutil
import torch
import torch.nn as nn

ARTIFACT_DIR = "deployment_artifacts"
OUTPUT_DIR = "web_export"


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
             
            #nn.GRUCell(4,4,False,None)
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(ARTIFACT_DIR, "config.json")) as f:
        config = json.load(f)

    class_names = config["class_names"]
    target_sr = config["target_sr"]
    max_length = target_sr * config["max_seconds"]

    model = AudioCNNStudent(num_classes=len(class_names))
    state_dict = torch.load(
        os.path.join(ARTIFACT_DIR, "student_model.pth"), map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Fixed-length dummy input matching the preprocessing pipeline
    # (waveform is always trimmed/padded to max_length before inference).
    dummy = torch.randn(1, max_length, dtype=torch.float32)

    onnx_path = os.path.join(OUTPUT_DIR, "student_model.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    print(f"Saved ONNX model -> {onnx_path}")

    shutil.copy(
        os.path.join(ARTIFACT_DIR, "config.json"),
        os.path.join(OUTPUT_DIR, "config.json"),
    )
    print(f"Copied config.json -> {OUTPUT_DIR}/config.json")
    print(
        f"Fixed input length expected by the web app: {max_length} samples "
        f"({config['max_seconds']}s @ {target_sr} Hz)"
    )
    print("\nUpload both files from web_export/ into the mobile app's Load Model section.")


if __name__ == "__main__":
    main()