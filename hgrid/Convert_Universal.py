import torch
from torch import nn
import torchvision.models as models

# =====================================================
# PILIH MODEL
# =====================================================
MODEL_NAME = "resnet18"
# Pilihan:
# "mobilenetv2"
# "mobilenetv3"
# "resnet18"

NUM_CLASSES = 18

# =====================================================
# MEMBUAT ARSITEKTUR
# =====================================================
if MODEL_NAME == "mobilenetv2":

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel,
        NUM_CLASSES
    )

    weight_path = "model_mobilenetv2.pth"
    output_path = "model_mobilenetv2.pt"

elif MODEL_NAME == "mobilenetv3":

    # Jika saat training memakai Large,
    # ganti menjadi mobilenet_v3_large()
    model = models.mobilenet_v3_small(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        NUM_CLASSES
    )

    weight_path = "model_mobilenetv3.pth"
    output_path = "model_mobilenetv3.pt"

elif MODEL_NAME == "resnet18":

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    weight_path = "model_resnet18.pth"
    output_path = "model_resnet18.pt"

else:
    raise Exception("Model tidak dikenali!")

# =====================================================
# LOAD WEIGHT
# =====================================================
model.load_state_dict(
    torch.load(weight_path, map_location="cpu")
)

model.eval()

# =====================================================
# TRACE
# =====================================================
example_input = torch.rand(1, 3, 128, 128)

with torch.no_grad():
    traced_model = torch.jit.trace(model, example_input)

# =====================================================
# SAVE UNTUK ANDROID
# =====================================================
traced_model._save_for_lite_interpreter(output_path)

print(f"✅ {output_path} berhasil dibuat!")