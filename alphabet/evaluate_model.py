import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from torchmetrics.classification import Accuracy
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# ======================
# MODEL LIST
# ======================
MODEL_LIST = [
    "mobilenetv2",
    "mobilenetv3",
    "resnet18",
    "ghostnet"
]


# ======================
# CONFIG
# ======================
DATA_DIR = "dataset_alphabet"

BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ======================
# DATASET
# ======================
test_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)


# ======================
# MODEL FUNCTION
# ======================
def get_model(model_name, num_classes):

    if model_name == "mobilenetv2":

        model = models.mobilenet_v2(
            weights=None
        )

        model.classifier[1] = nn.Linear(
            model.last_channel,
            num_classes
        )


    elif model_name == "mobilenetv3":

        model = models.mobilenet_v3_small(
            weights=None
        )

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes
        )


    elif model_name == "resnet18":

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )


    elif model_name == "ghostnet":

        import timm

        model = timm.create_model(
            "ghostnet_100",
            pretrained=False
        )

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )


    else:
        raise ValueError(
            "Model tidak dikenali"
        )

    return model


# ======================
# SAVE RESULTS
# ======================
results = []


# ======================
# EVALUATE ALL MODELS
# ======================
for MODEL_NAME in MODEL_LIST:

    print("\n==============================")
    print(f"Evaluating: {MODEL_NAME}")
    print("==============================")

    MODEL_PATH = (
        f"alphabet/model_alphabet_{MODEL_NAME}.pth"
    )


    # Load model
    model = get_model(
        MODEL_NAME,
        num_classes
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()


    all_preds = []
    all_labels = []


    acc = Accuracy(
        task="multiclass",
        num_classes=num_classes
    ).to(device)


    # ======================
    # TESTING
    # ======================
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(
                outputs,
                1
            )

            acc(preds, labels)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    # Accuracy desimal
    accuracy = acc.compute().item()


    print(
        f"\nTest Accuracy "
        f"({MODEL_NAME}): "
        f"{accuracy:.4f}"
    )


    results.append(
        (
            MODEL_NAME,
            accuracy
        )
    )


    # ======================
    # CONFUSION MATRIX
    # ======================
    cm = confusion_matrix(
        all_labels,
        all_preds
    )


    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4
        )
    )


    cm_normalized = (
        cm.astype("float")
        / cm.sum(axis=1)[:, np.newaxis]
    )


    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        f"Confusion Matrix "
        f"({MODEL_NAME})"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        f"alphabet_confusion_matrix_{MODEL_NAME}.png",
        dpi=300
    )

    plt.close()


# ======================
# SORT RESULTS
# ======================
results.sort(
    key=lambda x: x[1],
    reverse=True
)


# ======================
# PRINT FINAL TABLE
# ======================
print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

print(
    f"{'Model':<15} | Accuracy"
)

print("-" * 30)


for model_name, acc_val in results:

    print(
        f"{model_name:<15} | "
        f"{acc_val:.4f}"
    )


# ======================
# MINI BAR CHART
# ======================
models_name = [
    r[0].upper()
    for r in results
]

accuracies = [
    r[1]
    for r in results
]


plt.figure(figsize=(4, 3))

bars = plt.bar(
    models_name,
    accuracies
)


# Fokus ke range atas
plt.ylim(
    0.90,
    1.00
)


# Cleaner look
ax = plt.gca()

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)


plt.ylabel(
    "Accuracy",
    fontsize=8
)

plt.title(
    "ASL Model Comparison",
    fontsize=9
)

plt.xticks(
    fontsize=7
)

plt.yticks(
    fontsize=7
)


# Tampilkan nilai
for i, v in enumerate(
    accuracies
):

    plt.text(
        i,
        v + 0.002,
        f"{v:.4f}",
        ha="center",
        fontsize=7
    )


plt.tight_layout()

plt.savefig(
    "mini_asl_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nSaved:")
print(
    "mini_asl_model_comparison.png"
)