"""Unit tests for the dual-stream fusion classifier."""

import torch
import torch.nn as nn

from src.models.fusion_model import FusionModel
from src.models.fusion import FusionModel as SrcFusionModel


class DummyRGBBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * 4 * 4, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(start_dim=1))


class DummyFFTBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4 * 4, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(start_dim=1))


class RGBClassifierWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, 512),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FFTClassifierWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4, 256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def test_fusion_forward_with_feature_backbones() -> None:
    model = FusionModel(DummyRGBBackbone(), DummyFFTBackbone(), freeze_backbones=True)

    rgb_input = torch.randn(5, 3, 4, 4)
    fft_input = torch.randn(5, 1, 4, 4)
    logits = model(rgb_input, fft_input)

    assert logits.shape == (5, 2)


def test_fusion_forward_with_trimmed_classifier_wrappers() -> None:
    model = FusionModel(RGBClassifierWrapper(), FFTClassifierWrapper(), freeze_backbones=False)

    rgb_input = torch.randn(3, 3, 4, 4)
    fft_input = torch.randn(3, 1, 4, 4)
    logits = model(rgb_input, fft_input)

    assert logits.shape == (3, 2)


def test_freeze_backbones_only_affects_feature_extractors() -> None:
    model = FusionModel(DummyRGBBackbone(), DummyFFTBackbone(), freeze_backbones=True)

    assert all(not parameter.requires_grad for parameter in model.rgb_model.parameters())
    assert all(not parameter.requires_grad for parameter in model.fft_model.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_src_models_fusion_reexports_project_model() -> None:
    assert SrcFusionModel is FusionModel
