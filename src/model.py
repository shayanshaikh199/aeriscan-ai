"""
model.py
CNN model for Pneumonia detection
"""

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def get_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Convert first conv layer to 1-channel input
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Binary classifier
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model
