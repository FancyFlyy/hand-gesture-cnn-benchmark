import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from torchmetrics.classification import Accuracy

# ======================
# 1. TRANSFORM (KHUSUS HAGRID)
# ======================
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ======================
# 2. DATASET
# ======================
train_dataset = datasets.ImageFolder("dataset/train", transform=transform_train)
val_dataset = datasets.ImageFolder("dataset/val", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

# ======================
# 3. MODEL (PRETRAINED)
# ======================
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# ganti classifier
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ======================
# 4. LOSS & OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

# ======================
# 5. TRAINING
# ======================
num_epochs = 15

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

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

    print(f"Loss: {total_loss:.4f}")

# ======================
# 6. EVALUATION (VAL)
# ======================
acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        acc(preds, labels)

print(f"\nValidation Accuracy: {acc.compute()}")

# ======================
# 7. SAVE MODEL
# ======================
torch.save(model.state_dict(), "model_hagrid.pth")
print("Model saved!")

#0.389776349067688