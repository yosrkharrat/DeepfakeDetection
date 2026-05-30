"""Audio deepfake model loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.audio_stream import MFCCCNN, Wav2Vec2AudioClassifier


def _unwrap_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format for audio model.")


def load_audio_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load an audio model from a HF id or a local .pt/.pth checkpoint."""
    cp = str(checkpoint_path)
    resolved_device = torch.device(device)

    if Path(cp).exists() and (cp.endswith(".pth") or cp.endswith(".pt")):
        model = MFCCCNN().to(resolved_device)
        checkpoint = torch.load(cp, map_location="cpu")
        state_dict = _unwrap_checkpoint(checkpoint)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()
        return model

    model = Wav2Vec2AudioClassifier.from_pretrained(cp)
    model.to(resolved_device)
    model.eval()
    return model
