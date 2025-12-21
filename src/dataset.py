"""
dataset.py
Loads the Chest X-Ray Pneumonia dataset (grayscale)
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 224
BATCH_SIZE = 16

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 🔥 FIX
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 🔥 FIX
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
