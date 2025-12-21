"""
train.py
Trains Pneumonia classification model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders
from model import get_model
from tqdm import tqdm

EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    train_loader, val_loader, _ = get_loaders()
    model = get_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), "aeriscan_pneumonia.pth")
    print("✅ Model saved as aeriscan_pneumonia.pth")

if __name__ == "__main__":
    train()
