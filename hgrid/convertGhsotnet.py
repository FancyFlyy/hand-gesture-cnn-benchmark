import torch
from torch import nn
import timm

NUM_CLASSES = 18

# ==========================
# Buat arsitektur GhostNet
# ==========================
model = timm.create_model(
    "ghostnet_100",
    pretrained=False
)

model.classifier = nn.Linear(
    model.classifier.in_features,
    NUM_CLASSES
)

# ==========================
# Load Weight
# ==========================
model.load_state_dict(
    torch.load(
        "model_ghostnet.pth",
        map_location="cpu"
    )
)

model.eval()

# ==========================
# Convert ke TorchScript
# ==========================
example_input = torch.rand(1, 3, 128, 128)

with torch.no_grad():
    traced_model = torch.jit.trace(model, example_input)

# ==========================
# Save Android
# ==========================
traced_model._save_for_lite_interpreter(
    "model_ghostnet.pt"
)

print("✅ model_ghostnet.pt berhasil dibuat!")