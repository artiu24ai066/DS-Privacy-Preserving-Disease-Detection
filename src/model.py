"""
model.py
--------
MobileNetV2 classifier for binary pneumonia detection.

Architecture note — BatchNorm vs GroupNorm
------------------------------------------
The standard MobileNetV2 uses BatchNorm throughout its feature extractor.
Opacus DP-SGD requires per-sample gradients, which are incompatible with
BatchNorm because each sample's normalised value depends on its batch-mates.
ModuleValidator.fix() replaces all BatchNorm layers with GroupNorm, removing
the running_mean, running_var, and num_batches_tracked buffers.
This is applied to ALL experiments (E1–E4b) so architectures are identical.

Trainable parameter strategy
-----------------------------
Freezing the entire backbone and training only the 2,562-param classifier head
creates a catastrophic signal-to-noise ratio under DP-SGD:
  - The DP noise magnitude is calibrated to max_grad_norm=1.0
  - With only the classifier head trainable, the useful gradient signal
    is tiny compared to the injected noise
  - Result: the model collapses to predicting only PNEUMONIA (the majority class)

Fix: unfreeze the last `unfreeze_last_n` InvertedResidual blocks of MobileNetV2
(features[17] and features[18] when unfreeze_last_n=2). This adds ~32K more
trainable parameters, giving the DP optimiser enough signal to learn.

Trainable parameters:
  unfreeze_last_n=0  →   2,562 (classifier head only)     ← collapses with DP
  unfreeze_last_n=2  →  34,658 (head + last 2 feature blocks) ← works with DP

All experiments use unfreeze_last_n=2 for fair comparison.
The backbone blocks that remain frozen are features[0]–features[16].
"""

import copy
import torch
import torch.nn as nn
from torchvision import models


def build_model(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    unfreeze_last_n: int = 2,
) -> nn.Module:
    """
    Returns a DP-compatible MobileNetV2 with pretrained ImageNet weights.

    BatchNorm → GroupNorm replacement is applied unconditionally via
    ModuleValidator.fix() so that E1 and E2/E3/E4 share the same
    architecture and state-dict structure.

    Args:
        num_classes      : output classes (2 for NORMAL/PNEUMONIA)
        freeze_backbone  : if True, freeze feature extractor layers
                           (subject to unfreeze_last_n override)
        unfreeze_last_n  : number of MobileNetV2 InvertedResidual blocks
                           at the END of features[] to leave trainable even
                           when freeze_backbone=True.
                           Set to 0 for full freeze (not recommended with DP).
                           Set to 2 for the recommended configuration.
    """
    try:
        from opacus.validators import ModuleValidator
    except ImportError:
        raise ImportError("Opacus is required. Install with: pip install opacus")

    # Load ImageNet-pretrained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Replace default (1280→1000) classifier with binary head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes),
    )

    # Replace BatchNorm with GroupNorm for Opacus compatibility.
    # Applied to ALL experiments so architectures are identical.
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise RuntimeError(
            f"build_model: model is not Opacus-compatible after fix(): {errors}"
        )

    if freeze_backbone:
        # Freeze all feature extractor layers first
        for param in model.features.parameters():
            param.requires_grad = False

        # Then selectively unfreeze the last `unfreeze_last_n` blocks
        # MobileNetV2 features has 19 blocks (indices 0–18)
        if unfreeze_last_n > 0:
            n_feature_blocks = len(model.features)
            unfreeze_from = max(0, n_feature_blocks - unfreeze_last_n)
            for i in range(unfreeze_from, n_feature_blocks):
                for param in model.features[i].parameters():
                    param.requires_grad = True
                # Disable inplace activations in unfrozen blocks.
                # Inplace ops conflict with Opacus per-sample gradient hooks
                # (a RuntimeError: "Output of BackwardHookFunction is a view
                # and is being modified inplace").
                for submod in model.features[i].modules():
                    if hasattr(submod, "inplace"):
                        submod.inplace = False

    return model


def get_model_copy(model: nn.Module) -> nn.Module:
    """Returns a deep copy of a model (used by the federated server)."""
    return copy.deepcopy(model)


def count_parameters(model: nn.Module) -> int:
    """Returns number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for n in [0, 2]:
        m = build_model(unfreeze_last_n=n)
        print(f"unfreeze_last_n={n}: trainable={count_parameters(m):,}  "
              f"state-dict keys={len(m.state_dict())}")
    dummy = torch.randn(2, 3, 224, 224)
    out = build_model()(dummy)
    print(f"Forward pass output shape: {out.shape}")
