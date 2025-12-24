"""
Transfer Learning Model using ResNet18 for Face Expression Recognition
"""

import torch
from torch import nn
from torchvision import models


def create_transfer_model(num_classes: int):
    """
    Create a transfer learning model based on pretrained ResNet18.
    
    Args:
        num_classes: Number of output classes for classification
        
    Returns:
        Modified ResNet18 model with custom classification head
    """
    # Load pre-trained ResNet18
    model = models.resnet18(weights='DEFAULT')

    # Replace the output head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(256, num_classes)
    )
    return model

