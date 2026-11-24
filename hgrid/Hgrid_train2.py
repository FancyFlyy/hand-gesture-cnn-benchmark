import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from torchmetrics.classification import Accuracy


#  PILIH MODEL DI SINI

MODEL_NAME = "mobilenetv2"  
# opsi:
# "mobilenetv2"
# "mobilenetv3"
# "resnet18"
# "ghostnet"


# 1. FUNCTION MODEL

def get_model(model_name, num_classes):
    if model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "ghostnet":
        import timm
        model = timm.create_model("ghostnet_100", pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError("Model tidak dikenali")

    return model


# 2. TRANSFORM

transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# 3. DATASET

train_dataset = datasets.ImageFolder("dataset_hgrid/train", transform=transform_train)
val_dataset   = datasets.ImageFolder("dataset_hgrid/val", transform=transform_val)
test_dataset  = datasets.ImageFolder("dataset_hgrid/test", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

print("Classes:", train_dataset.classes)
print("Model:", MODEL_NAME)


# 4. MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(MODEL_NAME, num_classes)
model = model.to(device)


# 5. LOSS & OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 6. TRAINING + CURVE

num_epochs = 20

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    # TRAIN
    model.train()
    total_loss = 0

    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # VALIDATION
    model.eval()
    val_loss = 0
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            acc(preds, labels)

    val_loss /= len(val_loader)
    val_acc = acc.compute().item()

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


# 7. TEST (FINAL)

test_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_acc(preds, labels)

final_acc = test_acc.compute().item()

print(f"\n🔥 FINAL TEST ACCURACY ({MODEL_NAME}): {final_acc:.4f}")


# 8. SAVE MODEL

torch.save(model.state_dict(), f"model_{MODEL_NAME}.pth")


# 9. PLOT CURVE

plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss Curve ({MODEL_NAME})")
plt.legend()

plt.savefig(f"curve_{MODEL_NAME}.png")
plt.show()