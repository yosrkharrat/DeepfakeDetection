"""Audio extraction and preprocessing helpers.

Utilities to extract audio from video files (ffmpeg), load waveforms with
`librosa` when available, and compute mel-spectrograms for model inputs.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import librosa
except Exception:
    librosa = None


def extract_audio_from_video(video_path: str | Path, out_path: str | Path = "temp_audio.wav") -> str:
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(out_path),
        "-y",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_path)


def load_waveform(audio_path: str | Path, sr: int = 16000, duration: float | None = None) -> Tuple[np.ndarray, int]:
    if librosa is None:
        raise RuntimeError("librosa is required to load audio; install with `pip install librosa`")
    y, sr = librosa.load(str(audio_path), sr=sr, duration=duration)
    return y.astype(np.float32), sr


def compute_mel_spectrogram(waveform: np.ndarray, sr: int = 16000, n_mels: int = 128, n_fft: int = 1024, hop_length: int = 512) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa is required to compute mel spectrograms; install with `pip install librosa`")
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalize to 0..1
    S_norm = (S_db - S_db.min()) / max(1e-6, (S_db.max() - S_db.min()))
    return S_norm.astype(np.float32)
