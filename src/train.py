"""
train.py
Laptop-safe training for Pneumonia detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import get_model
from dataset import get_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3               # 🔽 Reduced
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "models/aeriscan_pneumonia.pth"

def train():
    train_loader, val_loader, _ = get_loaders()

    model = get_model().to(DEVICE)

    # 🔒 Freeze backbone (massive speed + stability boost)
    for param in model.parameters():
        param.requires_grad = False

    # Only train classifier head
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # 💾 SAVE CHECKPOINT EACH EPOCH
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"[✓] Checkpoint saved after epoch {epoch+1}")

    print("✅ Training complete")

if __name__ == "__main__":
    train()
