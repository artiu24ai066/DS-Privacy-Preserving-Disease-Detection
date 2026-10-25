"""
model.py
--------
MobileNetV2 classifier for binary pneumonia detection.
Uses ImageNet-pretrained weights with a replaced classifier head.
"""

import copy
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Returns a MobileNetV2 with pretrained ImageNet weights and a custom head.

    Args:
        num_classes     : number of output classes (2 for NORMAL / PNEUMONIA)
        freeze_backbone : if True, freeze all layers except the classifier head
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Replace the default classifier (1280 -> 1000) with our binary head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes),
    )

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    return model


def get_model_copy(model: nn.Module) -> nn.Module:
    """Returns a deep copy of a model (used by the federated server)."""
    return copy.deepcopy(model)


def count_parameters(model: nn.Module) -> int:
    """Returns number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model()
    print(f"Trainable parameters: {count_parameters(m):,}")
