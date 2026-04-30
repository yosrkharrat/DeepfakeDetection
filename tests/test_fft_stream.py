"""Unit tests for FFT preprocessing and checkpoint compatibility helpers."""

import torch
from unittest.mock import patch

from scripts.train_fusion import load_checkpoint
from src.models.fft_stream import (
    FFTOnlyClassifier,
    FFTStreamCNN,
    compute_fft_magnitude,
    normalize_fft_checkpoint_state_dict,
)


def test_compute_fft_magnitude_fft2_shape_and_range() -> None:
    image = torch.rand(2, 3, 224, 224)
    spectrum = compute_fft_magnitude(image, use_rfft=False)

    assert spectrum.shape == (2, 1, 224, 224)
    assert float(spectrum.min()) >= 0.0
    assert float(spectrum.max()) <= 1.0


def test_compute_fft_magnitude_rfft_shape_and_range() -> None:
    image = torch.rand(2, 3, 224, 224)
    spectrum = compute_fft_magnitude(image, use_rfft=True)

    assert spectrum.shape == (2, 1, 224, 113)
    assert float(spectrum.min()) >= 0.0
    assert float(spectrum.max()) <= 1.0


def test_fft_stream_accepts_raw_rgb_with_rfft_enabled() -> None:
    model = FFTStreamCNN(use_rfft=True)
    features = model(torch.rand(2, 3, 224, 224))

    assert features.shape == (2, 256)


def test_normalize_fft_checkpoint_state_dict_unwraps_classifier_checkpoint() -> None:
    wrapper = FFTOnlyClassifier()
    normalized_state, inferred_rfft = normalize_fft_checkpoint_state_dict(wrapper.state_dict())

    assert inferred_rfft is True
    assert all(not key.startswith("backbone.") for key in normalized_state)
    assert all(not key.startswith("classifier.") for key in normalized_state)
    assert "stem.0.weight" in normalized_state


def test_load_checkpoint_accepts_fft_classifier_checkpoint_under_strict_load() -> None:
    checkpoint = FFTOnlyClassifier().state_dict()

    with patch("scripts.train_fusion.load_serialized_checkpoint", return_value=checkpoint):
        model = FFTStreamCNN()
        load_checkpoint(model, "fft_classifier.pt", strict=True)

    assert model.uses_rfft is True
