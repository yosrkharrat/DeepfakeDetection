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
from src.models.fft_stream import FFTStreamCNN, FFTOnlyClassifier
from src.models.fusion_model import FusionModel
from src.models.rgb_stream import RGBStreamResNet, RGBOnlyClassifier
from src.models.audio_stream import Wav2Vec2AudioClassifier, MFCCCNN
from src.api.utils import audio_utils
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

    try:
        model.load_state_dict(_strip_module_prefix(checkpoint), strict=strict)
        return
    except RuntimeError:
        # Attempt a heuristic remapping for older checkpoints that saved
        # unnamed/sequential modules (e.g. keys like "0.0.1.0...", "3.weight").
        stripped = _strip_module_prefix(checkpoint)
        model_keys = list(model.state_dict().keys())

        # Build mapping by matching suffixes: remove the first numeric/anonymous
        # prefix from the checkpoint key and find a single model key that endswith
        # that suffix. This works for checkpoints that were saved from a
        # sequential wrapper around the FusionModel.
        mapped: dict[str, torch.Tensor] = {}
        for ck_key, tensor in stripped.items():
            parts = ck_key.split(".", 1)
            suffix = parts[1] if len(parts) > 1 else parts[0]
            matches = [mk for mk in model_keys if mk.endswith(suffix)]
            if len(matches) == 1:
                mapped[matches[0]] = tensor

        if not mapped:
            # Nothing to map — re-raise original error for visibility
            model.load_state_dict(_strip_module_prefix(checkpoint), strict=strict)

        # Load mapped parameters non-strictly so missing submodules are left
        # at their default initialization.
        # Try to improve mapping for ambiguous single-token keys by matching
        # tensor shapes when possible (e.g., keys like "3.weight" ->
        # "classifier.0.weight").
        remaining_ck = {k: v for k, v in stripped.items() if k not in mapped.values()}
        # Build reverse lookup of model key shapes
        model_state = model.state_dict()
        shape_map: dict[tuple[int, ...], list[str]] = {}
        for mk, mv in model_state.items():
            shape_map.setdefault(tuple(mv.shape), []).append(mk)

        for ck_key, tensor in stripped.items():
            parts = ck_key.split(".", 1)
            if len(parts) == 1 or parts[0].isdigit():
                # single-token or numeric-prefix keys
                shp = tuple(tensor.shape)
                candidates = shape_map.get(shp, [])
                # prefer candidates that are not yet mapped
                candidates = [c for c in candidates if c not in mapped]
                if len(candidates) == 1:
                    mapped[candidates[0]] = tensor

        model.load_state_dict(mapped, strict=False)


def _image_to_tensor(image_hwc: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image_hwc).permute(2, 0, 1).contiguous().float()
    if image_hwc.dtype == np.uint8:
        return tensor / 255.0
    return tensor


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
    mode: str = "fusion",
) -> nn.Module:
    """Load a trained model checkpoint for inference.

    mode: one of 'fusion', 'rgb', 'fft', 'audio'
    """
    resolved_device = torch.device(device)

    if mode == "fusion":
        model = FusionModel(
            rgb_model=RGBStreamResNet(pretrained=False),
            fft_model=FFTStreamCNN(),
            freeze_backbones=True,
        ).to(resolved_device)
    elif mode == "rgb":
        model = RGBOnlyClassifier(pretrained=False).to(resolved_device)
    elif mode == "fft":
        model = FFTOnlyClassifier().to(resolved_device)
    elif mode == "audio":
        # Allow two kinds of audio checkpoints:
        # - a HuggingFace model id/name (string) -> load Wav2Vec2 wrapper
        # - a local .pth/.pt checkpoint matching MFCCCNN -> load weights
        cp = str(checkpoint_path)
        if Path(cp).exists() and (cp.endswith(".pth") or cp.endswith(".pt")):
            model = MFCCCNN().to(resolved_device)
            _load_checkpoint(model, checkpoint_path, strict=strict)
            model.eval()
            return model
        else:
            # Treat as HF model id
            wav = Wav2Vec2AudioClassifier.from_pretrained(cp)
            wav.to(resolved_device)
            wav.eval()
            return wav
    elif mode == "ensemble":
        # Not used here; ensemble loading is handled in from_checkpoint
        raise ValueError("Use from_checkpoint with mode='ensemble' to load ensemble models")
    else:
        raise ValueError("mode must be one of 'fusion', 'rgb', 'fft'")

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
        # model may be a FusionModel, a single-stream classifier, or a tuple
        # (rgb_model, fft_model) when using ensemble mode.
        self.mode = "fusion"
        if isinstance(model, tuple) and len(model) == 2:
            self.mode = "ensemble"
            self.rgb_model, self.fft_model = model
            self.rgb_model = self.rgb_model.to(self.device)
            self.fft_model = self.fft_model.to(self.device)
            self.rgb_model.eval()
            self.fft_model.eval()
        elif isinstance(model, tuple) and len(model) == 3:
            # (rgb_model, fft_model, audio_model)
            self.mode = "av_ensemble"
            self.rgb_model, self.fft_model, self.audio_model = model
            self.rgb_model = self.rgb_model.to(self.device)
            self.fft_model = self.fft_model.to(self.device)
            # audio_model may be a HF wrapper or a small nn.Module
            try:
                self.audio_model = self.audio_model.to(self.device)
            except Exception:
                pass
            self.rgb_model.eval()
            self.fft_model.eval()
            try:
                self.audio_model.eval()
            except Exception:
                pass
        else:
            self.model = model.to(self.device)
            self.model.eval()
        self.image_size = image_size
        self.min_face_size = min_face_size
        self.threshold = threshold
        self.transform = get_eval_augmentation(image_size)
        self.face_detector = FaceDetector(device=str(self.device))
        # Temperature for scaling logits -> probabilities. If a file
        # results/checkpoints/temperature.json exists it will be loaded
        # by the calibration script; default is 1.0 (no scaling).
        self.temperature = 1.0
        try:
            import json

            temp_path = Path("results/checkpoints/temperature.json")
            if temp_path.exists():
                with open(temp_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    t = float(data.get("temperature", 1.0))
                    if t > 0.0:
                        self.temperature = t
        except Exception:
            # Ignore failures to load temperature; keep default
            pass

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        image_size: int = 224,
        min_face_size: int = 64,
        threshold: float = 0.5,
        strict: bool = True,
        mode: str = "fusion",
    ) -> "FusionInferenceService":
        if mode == "ensemble":
            # Allow overriding the two checkpoints via environment variables
            # `RGB_CHECKPOINT` and `FFT_CHECKPOINT`. If not provided, both
            # models will attempt to load from `checkpoint_path`.
            import os

            rgb_ckpt = os.environ.get("RGB_CHECKPOINT") or str(checkpoint_path)
            fft_ckpt = os.environ.get("FFT_CHECKPOINT") or str(checkpoint_path)

            rgb_model = load_model(rgb_ckpt, device=device, strict=strict, mode="rgb")
            fft_model = load_model(fft_ckpt, device=device, strict=strict, mode="fft")
            return cls(
                model=(rgb_model, fft_model),
                device=device,
                image_size=image_size,
                min_face_size=min_face_size,
                threshold=threshold,
            )

        model = load_model(checkpoint_path, device=device, strict=strict, mode=mode)
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

        # Build FFT input batch from raw uint8 crops (no ImageNet normalization).
        # FFTStreamCNN expects either raw RGB in [0,1] or a precomputed spectrum.
        fft_tensors: list[torch.Tensor] = []
        for crop_bgr in crops_bgr:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            # _image_to_tensor will divide by 255 for uint8 input producing [0,1]
            fft_tensors.append(_image_to_tensor(crop_rgb))

        fft_batch = torch.stack(fft_tensors).to(self.device, non_blocking=True)

        with torch.no_grad():
            # Compute logits for all supported modes; apply temperature
            # scaling to logits before converting to probabilities.
            if self.mode == "ensemble":
                logits_rgb = self.rgb_model(rgb_batch)
                logits_fft = self.fft_model(fft_batch)
                # Average logits then apply temperature
                logits = (logits_rgb + logits_fft) / 2.0
            else:
                # Support fusion and single-stream models. Fusion models accept
                # (rgb_batch, fft_batch). Single-stream classifiers accept a
                # single tensor (rgb_batch or fft_batch).
                try:
                    logits = self.model(rgb_batch, fft_batch)
                except TypeError:
                    # Try RGB-only classifier
                    try:
                        logits = self.model(rgb_batch)
                    except TypeError:
                        # Fall back to FFT-only classifier
                        logits = self.model(fft_batch)

            # Apply temperature scaling (logits / T) before softmax
            T = float(self.temperature) if hasattr(self, "temperature") else 1.0
            if T <= 0.0:
                T = 1.0
            scaled_logits = logits / float(T)
            probabilities = probabilities_from_logits(scaled_logits).detach().cpu()

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

    def predict_audio_file(self, audio_path: str | Path) -> dict[str, Any]:
        """Run audio-only prediction on a WAV file and return probabilities.

        Returns a dict with keys `is_fake`, `confidence`, and `probabilities`.
        """
        # Ensure we have an audio model available
        if not hasattr(self, "audio_model"):
            raise RuntimeError("No audio model is loaded for this service.")

        # Load waveform
        waveform, sr = audio_utils.load_waveform(audio_path, sr=16000)

        # Use audio model's helper if available
        model = self.audio_model
        # Some wrappers provide `predict_logits`; otherwise try forward
        if hasattr(model, "predict_logits"):
            logits = model.predict_logits(waveform, sr=sr)
        else:
            # As a fallback, compute mel and run forward
            mel = audio_utils.compute_mel_spectrogram(waveform, sr=sr)
            mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                logits = model(mel_t).detach().cpu()

        # Apply temperature scaling
        T = float(self.temperature) if hasattr(self, "temperature") else 1.0
        if T <= 0.0:
            T = 1.0
        scaled = logits / float(T)
        probs = probabilities_from_logits(scaled).detach().cpu()
        p_fake = float(probs[0, 1].item())
        p_real = float(probs[0, 0].item())
        is_fake = p_fake >= self.threshold
        confidence = p_fake if is_fake else p_real
        return {"is_fake": bool(is_fake), "confidence": float(confidence), "probabilities": {"real": p_real, "fake": p_fake}}

    def predict_video_path(self, video_path: str | Path, weight_visual: float = 0.6, weight_audio: float = 0.4) -> dict[str, Any]:
        """Predict on a video file by combining visual and audio streams.

        - visual prediction: uses the first frame and existing face pipeline
        - audio prediction: extracts audio via ffmpeg and runs audio model

        Returns combined probabilities and per-stream outputs.
        """
        # Visual: extract first frame with cv2
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to read any frame from video: {video_path}")

        visual_out = self.predict_image_array(frame)
        # visual fake prob: highest face fake probability
        face_probs = [f["probabilities"]["fake"] for f in visual_out.get("faces", [])]
        p_fake_visual = max(face_probs) if face_probs else 0.0

        # Audio
        audio_temp = Path("temp_audio_from_video.wav")
        audio_path = audio_utils.extract_audio_from_video(video_path, out_path=audio_temp)
        audio_out = self.predict_audio_file(audio_path)
        p_fake_audio = audio_out["probabilities"]["fake"]

        # Combine weights
        total = float(weight_visual + weight_audio)
        if total <= 0.0:
            total = 1.0
        p_combined = (weight_visual * p_fake_visual + weight_audio * p_fake_audio) / total

        return {
            "p_fake_visual": float(p_fake_visual),
            "p_fake_audio": float(p_fake_audio),
            "p_fake_combined": float(p_combined),
            "visual": visual_out,
            "audio": audio_out,
        }
