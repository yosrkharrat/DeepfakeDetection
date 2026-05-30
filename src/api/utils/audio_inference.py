"""Helpers for running audio inference on video files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.api.utils import audio_utils


def predict_audio_from_video(service, video_path: str | Path) -> dict:
    """Extract audio from a video file and run the service audio model."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        audio_path = audio_utils.extract_audio_from_video(video_path, out_path=tmp_path)
        return service.predict_audio_file(audio_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
