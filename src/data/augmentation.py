"""Albumentations pipelines for training and evaluation."""

from __future__ import annotations

from typing import Any


def _require_albumentations() -> Any:
    try:
        import albumentations as A
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "Albumentations is required for augmentation. Install with: pip install albumentations"
        ) from exc
    return A


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_compression_transform(A: Any):
    """Support both old and new Albumentations ImageCompression signatures."""
    try:
        return A.ImageCompression(quality_range=(60, 100), p=0.3)
    except TypeError:
        return A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3)


def get_train_augmentation(image_size: int = 224):
    """Training augmentation policy aligned with project spec."""
    A = _require_albumentations()
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.4),
            _image_compression_transform(A),
            A.RandomBrightnessContrast(p=0.4),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_augmentation(image_size: int = 224):
    """Deterministic preprocessing used for validation/test."""
    A = _require_albumentations()
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
