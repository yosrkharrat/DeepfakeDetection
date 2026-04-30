"""Unit tests for RGB backbone checkpoint compatibility."""

import numpy as np
import torch
from unittest.mock import patch

from scripts.train_fusion import load_checkpoint
from src.models.rgb_stream import (
    RGBOnlyClassifier,
    RGBStreamResNet,
    normalize_rgb_checkpoint_state_dict,
)


def test_rgb_stream_outputs_512d_features() -> None:
    model = RGBStreamResNet(pretrained=False)
    features = model(torch.rand(2, 3, 224, 224))

    assert features.shape == (2, 512)


def test_normalize_rgb_checkpoint_state_dict_unwraps_local_classifier_checkpoint() -> None:
    wrapper = RGBOnlyClassifier(pretrained=False)
    normalized = normalize_rgb_checkpoint_state_dict(wrapper.state_dict())

    RGBStreamResNet(pretrained=False).load_state_dict(normalized, strict=True)
    assert all(not key.startswith("backbone.backbone.") for key in normalized)
    assert all(not key.startswith("classifier.") for key in normalized)


def test_normalize_rgb_checkpoint_state_dict_drops_external_head() -> None:
    model = RGBStreamResNet(pretrained=False)
    kaggle_like = dict(model.state_dict())
    kaggle_like["head.0.weight"] = torch.randn(256, 512)
    kaggle_like["head.0.bias"] = torch.randn(256)

    normalized = normalize_rgb_checkpoint_state_dict(kaggle_like)
    RGBStreamResNet(pretrained=False).load_state_dict(normalized, strict=True)
    assert all(not key.startswith("head.") for key in normalized)


def test_load_checkpoint_accepts_kaggle_style_rgb_training_checkpoint() -> None:
    wrapper = RGBOnlyClassifier(pretrained=False)
    checkpoint = {
        "model_state_dict": wrapper.state_dict(),
        "val_acc": np.float32(0.95),
        "cfg": {"dropout": 0.3},
    }

    with patch("scripts.train_fusion.load_serialized_checkpoint", return_value=checkpoint):
        model = RGBStreamResNet(pretrained=False)
        load_checkpoint(model, "rgb_training_checkpoint.pt", strict=True)
