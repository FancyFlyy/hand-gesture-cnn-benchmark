import numpy as np                  # library untuk operasi array (dipakai saat tampilkan gambar)
import matplotlib.pyplot as plt    # untuk menampilkan gambar / grafik

import torch                       # core PyTorch
from torch import nn, optim        # nn = neural network, optim = optimizer
from torch.utils.data import DataLoader  # untuk ambil data per batch
from tqdm import tqdm              # progress bar saat training

import torchvision                 # library tambahan untuk computer vision
import torchvision.datasets as datasets  # untuk load dataset gambar
import torchvision.transforms as transforms  # preprocessing gambar
import torch.nn.functional as F   # fungsi aktivasi seperti ReLU

from torchmetrics.classification import Accuracy  # hitung akurasi



# ======================
# 1. TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # ubah ukuran semua gambar jadi 64x64
    transforms.ToTensor()          # ubah gambar jadi tensor (format PyTorch)
])

# ======================
# 2. LOAD DATASET
# ======================
train_dataset = datasets.ImageFolder(
    root="dataset/train",          # ambil data dari folder train
    transform=transform            # apply transform ke semua gambar
)

test_dataset = datasets.ImageFolder(
    root="dataset/val",            # ambil data dari folder validation
    transform=transform
)

batch_size = 32                    # jumlah data dalam 1 batch

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# shuffle=True → data diacak supaya model tidak menghafal urutan

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("Classes:", train_dataset.classes)
# menampilkan nama folder (label), contoh: ['thumbs_up', 'thumbs_down']

# ======================
# 3. VISUALISASI DATA
# ======================
def imshow(img):
    npimg = img.numpy()   # ubah tensor ke numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  
    # ubah format (channel, height, width) → (height, width, channel)
    plt.show()            # tampilkan gambar

dataiter = iter(train_loader)        # ambil iterator dari dataloader
images, labels = next(dataiter)      # ambil 1 batch data

imshow(torchvision.utils.make_grid(images))
# gabungkan banyak gambar jadi grid lalu tampilkan

# ======================
# 4. MODEL CNN
# ======================
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        # layer conv pertama (RGB=3 channel → 16 feature map)

        self.pool = nn.MaxPool2d(2, 2)
        # memperkecil ukuran gambar (downsampling)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # layer conv kedua (16 → 32 feature map)

        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)
        # fully connected layer (hasil akhir ke jumlah kelas)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # conv1 + aktivasi ReLU
        x = self.pool(x)           # pooling

        x = F.relu(self.conv2(x))  # conv2 + ReLU
        x = self.pool(x)           # pooling lagi

        x = x.reshape(x.shape[0], -1)  # flatten (ubah ke 1D)
        x = self.fc1(x)                # klasifikasi

        return x                      # output (logits)

# ======================
# 5. SET DEVICE
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
# pakai GPU kalau ada, kalau tidak pakai CPU

num_classes = len(train_dataset.classes)
# jumlah kelas otomatis dari folder

model = CNN(in_channels=3, num_classes=num_classes).to(device)
# in_channels=3 karena gambar RGB

print(model)  # tampilkan struktur model

# ======================
# 6. LOSS & OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss()
# fungsi loss untuk klasifikasi

optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer untuk update weight model

# ======================
# 7. TRAINING
# ======================
num_epochs = 5

for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")

    for data, targets in tqdm(train_loader):
        data = data.to(device)        # kirim data ke GPU/CPU
        targets = targets.to(device)

        outputs = model(data)         # forward pass (prediksi)
        loss = criterion(outputs, targets)  # hitung error

        optimizer.zero_grad()  # reset grad sebelumnya
        loss.backward()        # backpropagation (hitung grad)
        optimizer.step()       # update weight

# ======================
# 8. EVALUATION
# ======================
acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

model.eval()   # mode evaluasi (matikan dropout dll)
with torch.no_grad():  # tidak hitung grad (lebih cepat)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)       # prediksi
        _, preds = torch.max(outputs, 1)
        # ambil kelas dengan nilai tertinggi

        acc(preds, labels)            # hitung akurasi

print(f"Test Accuracy: {acc.compute()}")
# tampilkan hasil akhir akurasi

# ======================
# 9. SAVE MODEL
# ======================
torch.save(model.state_dict(), "model.pth")
# simpan model ke file

print("Model saved!")



#Test Accuracy: 0.45527157187461853