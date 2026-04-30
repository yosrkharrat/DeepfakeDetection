"""Inference helpers for the trained dual-stream fusion model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import cv2
except ImportError:  # pragma: no cover - runtime env dependent
    cv2 = None

from src.data.augmentation import get_eval_augmentation
from src.data.face_detector import FaceDetector
from src.models.fft_stream import FFTStreamCNN
from src.models.fusion_model import FusionModel
from src.models.rgb_stream import RGBStreamResNet
from src.utils.metrics import probabilities_from_logits


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not all(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _load_checkpoint(model: nn.Module, checkpoint_path: str | Path, strict: bool = True) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}.")

    model.load_state_dict(_strip_module_prefix(checkpoint), strict=strict)


def _image_to_tensor(image_hwc: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image_hwc).permute(2, 0, 1).contiguous().float()
    if image_hwc.dtype == np.uint8:
        return tensor / 255.0
    return tensor


def load_fusion_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> FusionModel:
    """Load a trained fusion model checkpoint for inference."""
    resolved_device = torch.device(device)
    model = FusionModel(
        rgb_model=RGBStreamResNet(pretrained=False),
        fft_model=FFTStreamCNN(),
        freeze_backbones=True,
    ).to(resolved_device)
    _load_checkpoint(model, checkpoint_path, strict=strict)
    model.eval()
    return model


class FusionInferenceService:
    """Face-level inference wrapper for the fusion model."""

    def __init__(
        self,
        model: FusionModel,
        device: str | torch.device = "cpu",
        image_size: int = 224,
        min_face_size: int = 64,
        threshold: float = 0.5,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for inference. Install with: pip install opencv-python")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in the [0.0, 1.0] interval.")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.image_size = image_size
        self.min_face_size = min_face_size
        self.threshold = threshold
        self.transform = get_eval_augmentation(image_size)
        self.face_detector = FaceDetector(device=str(self.device))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        image_size: int = 224,
        min_face_size: int = 64,
        threshold: float = 0.5,
        strict: bool = True,
    ) -> "FusionInferenceService":
        model = load_fusion_model(checkpoint_path, device=device, strict=strict)
        return cls(
            model=model,
            device=device,
            image_size=image_size,
            min_face_size=min_face_size,
            threshold=threshold,
        )

    def _prepare_tensor(self, crop_bgr: np.ndarray) -> torch.Tensor:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=crop_rgb)
        image_hwc = transformed["image"] if isinstance(transformed, dict) else transformed
        return _image_to_tensor(image_hwc)

    def _predict_crops(
        self,
        crops_bgr: list[np.ndarray],
        boxes: list[tuple[int, int, int, int]],
    ) -> list[dict[str, Any]]:
        rgb_tensors: list[torch.Tensor] = []
        for crop_bgr in crops_bgr:
            rgb_tensors.append(self._prepare_tensor(crop_bgr))

        rgb_batch = torch.stack(rgb_tensors).to(self.device, non_blocking=True)

        with torch.no_grad():
            logits = self.model(rgb_batch, rgb_batch)
            probabilities = probabilities_from_logits(logits).detach().cpu()

        results: list[dict[str, Any]] = []
        for box, probs in zip(boxes, probabilities):
            real_probability = float(probs[0].item())
            fake_probability = float(probs[1].item())
            is_fake = fake_probability >= self.threshold
            confidence = fake_probability if is_fake else real_probability

            results.append(
                {
                    "bbox": {
                        "x": int(box[0]),
                        "y": int(box[1]),
                        "w": int(box[2]),
                        "h": int(box[3]),
                    },
                    "is_fake": bool(is_fake),
                    "confidence": float(confidence),
                    "probabilities": {
                        "real": real_probability,
                        "fake": fake_probability,
                    },
                }
            )

        return results

    def predict_image_array(self, image_bgr: np.ndarray) -> dict[str, Any]:
        """Run detection + fusion inference on an in-memory BGR image."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected a BGR image with shape (H, W, 3), received {tuple(image_bgr.shape)}.")

        boxes = self.face_detector.detect_boxes(image_bgr)
        used_full_frame_fallback = False

        if not boxes:
            height, width = image_bgr.shape[:2]
            boxes = [(0, 0, width, height)]
            used_full_frame_fallback = True

        crops: list[np.ndarray] = []
        valid_boxes: list[tuple[int, int, int, int]] = []
        for x, y, w, h in boxes:
            crop = image_bgr[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            crops.append(crop)
            valid_boxes.append((x, y, w, h))

        if not crops:
            raise RuntimeError("Unable to extract any valid face crop for inference.")

        face_results = self._predict_crops(crops, valid_boxes)
        best_face = max(face_results, key=lambda result: result["probabilities"]["fake"])

        return {
            "is_fake": best_face["is_fake"],
            "confidence": best_face["confidence"],
            "used_full_frame_fallback": used_full_frame_fallback,
            "num_faces": len(face_results),
            "faces": face_results,
        }

    def predict_path(self, image_path: str | Path) -> dict[str, Any]:
        """Load an image from disk and return the structured prediction."""
        if cv2 is None:
            raise RuntimeError("OpenCV is required for inference. Install with: pip install opencv-python")

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        return self.predict_image_array(image)
