"""
dataset.py
Pneumonia dataset loader (CPU safe)
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 224
BATCH_SIZE = 8   # 🔽 was 16

def get_transforms(train=True):
    base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]

    if train:
        base.insert(2, transforms.RandomHorizontalFlip())

    return transforms.Compose(base)

def load_datasets(base_dir="data/chest_xray"):
    train_data = datasets.ImageFolder(
        os.path.join(base_dir, "train"),
        transform=get_transforms(train=True)
    )
    val_data = datasets.ImageFolder(
        os.path.join(base_dir, "val"),
        transform=get_transforms(train=False)
    )
    test_data = datasets.ImageFolder(
        os.path.join(base_dir, "test"),
        transform=get_transforms(train=False)
    )
    return train_data, val_data, test_data

def get_loaders():
    train_ds, val_ds, test_ds = load_datasets()
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    )
