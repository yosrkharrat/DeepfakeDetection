"""RGB stream backbone for deepfake detection."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

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


def _load_backbone(backbone_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    try:
        tv_models = importlib.import_module("torchvision.models")
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for RGBStreamResNet. Install it with: pip install torchvision"
        ) from exc

    name = backbone_name.lower().replace("-", "_")

    if name == "efficientnet_b4":
        efficientnet_b4 = getattr(tv_models, "efficientnet_b4")
        weights_enum = getattr(tv_models, "EfficientNet_B4_Weights", None)
        if weights_enum is not None:
            weights = weights_enum.DEFAULT if pretrained else None
            backbone = efficientnet_b4(weights=weights)
        else:
            backbone = efficientnet_b4(pretrained=pretrained)
        feature_dim = backbone.classifier[1].in_features
        trunk = nn.Sequential(*list(backbone.children())[:-1])
        return trunk, int(feature_dim)

    if name == "convnext_tiny":
        convnext_tiny = getattr(tv_models, "convnext_tiny")
        weights_enum = getattr(tv_models, "ConvNeXt_Tiny_Weights", None)
        if weights_enum is not None:
            weights = weights_enum.DEFAULT if pretrained else None
            backbone = convnext_tiny(weights=weights)
        else:
            backbone = convnext_tiny(pretrained=pretrained)
        feature_dim = backbone.classifier[-1].in_features
        trunk = nn.Sequential(backbone.features, backbone.avgpool)
        return trunk, int(feature_dim)

    if name == "resnet50":
        resnet50 = getattr(tv_models, "resnet50")
        weights_enum = getattr(tv_models, "ResNet50_Weights", None)
        if weights_enum is not None:
            weights = weights_enum.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
        else:
            backbone = resnet50(pretrained=pretrained)
        feature_dim = backbone.fc.in_features
        trunk = nn.Sequential(*list(backbone.children())[:-1])
        return trunk, int(feature_dim)

    raise ValueError(f"Unsupported backbone_name: {backbone_name}")


class RGBStreamResNet(nn.Module):
    """RGB backbone that returns global feature vectors."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2, backbone_name: str = "efficientnet_b4"):
        super().__init__()
        _ = dropout  # Kept for API compatibility with earlier code.

        self.backbone_name = backbone_name
        self.backbone, self.feature_dim = _load_backbone(backbone_name, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)
        return features


class RGBOnlyClassifier(nn.Module):
    """Optional RGB-only baseline classifier for ablations."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2, backbone_name: str = "efficientnet_b4"):
        super().__init__()
        self.backbone = RGBStreamResNet(pretrained=pretrained, dropout=dropout, backbone_name=backbone_name)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class ResNet18Classifier(nn.Module):
    """ResNet-18 backbone with a 2-class head.

    Matches the checkpoint layout saved by rgb-training notebooks:
        backbone = nn.Sequential(*resnet18.children()[:-1])
        head     = Sequential(Flatten, Dropout, Linear(512,256), ReLU, Dropout, Linear(256,2))
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        try:
            tv_models = importlib.import_module("torchvision.models")
        except ImportError as exc:
            raise ImportError("torchvision is required for ResNet18Classifier") from exc
        resnet = tv_models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


if __name__ == "__main__":
    print("=== RGB Stream sanity check ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dummy_rgb = torch.rand(4, 3, 224, 224).to(device)

    model = RGBStreamResNet(pretrained=False).to(device)
    features = model(dummy_rgb)
    print(f"Feature vector shape: {features.shape}")  # (4, 1792)

    clf = RGBOnlyClassifier(pretrained=False).to(device)
    logits = clf(dummy_rgb)
    print(f"Classifier logits shape: {logits.shape}")  # (4, 2)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nRGBStreamResNet trainable params: {n_params:,}")

    print("\nAll checks passed.")
