"""RGB stream backbone for deepfake detection."""

from __future__ import annotations

import importlib
from typing import Dict

import torch
import torch.nn as nn


def normalize_rgb_checkpoint_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize wrapper checkpoints so they load into RGBStreamResNet."""
    if any(key.startswith("backbone.backbone.") for key in state_dict):
        return {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }

    if any(key.startswith("head.") for key in state_dict) and any(
        key.startswith("backbone.") for key in state_dict
    ):
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }

    return state_dict


class RGBStreamResNet(nn.Module):
    """ResNet-18 body that returns 512-d global features."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        _ = dropout  # Kept for API compatibility with earlier code.

        try:
            tv_models = importlib.import_module("torchvision.models")
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for RGBStreamResNet. Install it with: pip install torchvision"
            ) from exc

        resnet18 = getattr(tv_models, "resnet18")
        weights_enum = getattr(tv_models, "ResNet18_Weights", None)

        if weights_enum is not None:
            weights = weights_enum.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(pretrained=pretrained)

        # Keep the global average pooling output and drop the 1000-class head.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return torch.flatten(features, start_dim=1)


class RGBOnlyClassifier(nn.Module):
    """Optional RGB-only baseline classifier for ablations."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = RGBStreamResNet(pretrained=pretrained, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


if __name__ == "__main__":
    print("=== RGB Stream sanity check ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dummy_rgb = torch.rand(4, 3, 224, 224).to(device)

    model = RGBStreamResNet(pretrained=False).to(device)
    features = model(dummy_rgb)
    print(f"Feature vector shape: {features.shape}")  # (4, 512)

    clf = RGBOnlyClassifier(pretrained=False).to(device)
    logits = clf(dummy_rgb)
    print(f"Classifier logits shape: {logits.shape}")  # (4, 2)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nRGBStreamResNet trainable params: {n_params:,}")

    print("\nAll checks passed.")
