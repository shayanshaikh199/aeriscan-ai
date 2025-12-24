import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ================= CONFIG =================
DATA_ROOT = "data/chest_xray"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
MODEL_DIR = "models"

BATCH_SIZE = 8        # Laptop-safe
EPOCHS = 6            # Keep <= 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= TRANSFORMS =================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ================= DATASETS =================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("[INFO] Classes:", train_dataset.classes)
print("[INFO] Training samples:", len(train_dataset))
print("[INFO] Validation samples:", len(val_dataset))

# ================= CLASS WEIGHTS =================
labels = [label for _, label in train_dataset.samples]
neg = labels.count(0)
pos = labels.count(1)

weight_for_0 = 1.0
weight_for_1 = neg / pos
class_weights = torch.tensor([weight_for_0, weight_for_1]).to(DEVICE)

# ================= MODEL =================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 1-channel input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Binary output
model.fc = nn.Linear(512, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= TRAINING =================
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # ===== Validation =====
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total * 100

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f}")
    print(f"Val Acc:    {val_acc:.2f}%")

    # Save checkpoint
    torch.save(
        model.state_dict(),
        f"{MODEL_DIR}/aeriscan_epoch_{epoch+1}.pth"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            model.state_dict(),
            f"{MODEL_DIR}/aeriscan_best.pth"
        )
        print("[✓] Best model updated")

print("\n[✓] Training complete")
print("[✓] Best model saved as models/aeriscan_best.pth")
