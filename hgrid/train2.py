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
# 1. TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ======================
# 2. DATASET
# ======================
train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

# ======================
# 3. MODEL (VERSI AMAN)
# ======================
model = models.mobilenet_v2(weights=None)  # ❗ tidak pakai pretrained dulu

model.classifier[1] = nn.Linear(model.last_channel, num_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ======================
# 4. LOSS & OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======================
# 5. TRAINING
# ======================
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")

    model.train()
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ======================
# 6. EVALUATION
# ======================
acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        acc(preds, labels)

print(f"Test Accuracy: {acc.compute()}")

# ======================
# 7. SAVE MODEL
# ======================
torch.save(model.state_dict(), "model.pth")
print("Model saved!")


#Test Accuracy: 0.9456868767738342