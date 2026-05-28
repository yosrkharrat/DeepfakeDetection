"""Audio models and wrappers for AV-deepfake detection.

Provides two options:
- `Wav2Vec2AudioClassifier` (optional; requires `transformers`) — wraps
  `Wav2Vec2ForSequenceClassification` and exposes `predict_logits(waveform, sr)`.
- `MFCCCNN` — small CNN that accepts mel-spectrogram inputs and exposes
  `predict_logits(waveform, sr)` for convenience.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import (
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2Processor,
        Wav2Vec2FeatureExtractor,
    )
except Exception:
    Wav2Vec2ForSequenceClassification = None
    Wav2Vec2Processor = None
    Wav2Vec2FeatureExtractor = None

from src.api.utils import audio_utils


class Wav2Vec2AudioClassifier(nn.Module):
    """Wrapper around a HuggingFace Wav2Vec2 model for sequence classification.

    Example usage:
        model = Wav2Vec2AudioClassifier.from_pretrained('facebook/wav2vec2-base')
        logits = model.predict_logits(waveform, sr=16000)
    """

    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        # `processor` may be a full Wav2Vec2Processor (feature_extractor + tokenizer)
        # or None. Keep a reference to the available objects.
        self.processor = processor
        self.feature_extractor = None
        if processor is None and Wav2Vec2FeatureExtractor is not None:
            try:
                # try to load a compatible feature extractor from the model
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model.name_or_path)
            except Exception:
                self.feature_extractor = None
        self.is_wav2vec = True

    @classmethod
    def from_pretrained(cls, name: str = "facebook/wav2vec2-base", num_labels: int = 2):
        if Wav2Vec2ForSequenceClassification is None:
            raise RuntimeError("transformers not available; install transformers to use Wav2Vec2 models")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(name, num_labels=num_labels)
        # Some HF repos provide a processor, others only a feature extractor.
        processor = None
        feature_extractor = None
        try:
            processor = Wav2Vec2Processor.from_pretrained(name)
        except Exception:
            try:
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
            except Exception:
                feature_extractor = None
        try:
            model.freeze_feature_encoder()
        except Exception:
            pass
        inst = cls(model, processor)
        # if we found a standalone feature extractor attach it
        if getattr(inst, "feature_extractor", None) is None and feature_extractor is not None:
            inst.feature_extractor = feature_extractor
        return inst

    def predict_logits(self, waveform: np.ndarray, sr: int = 16000) -> torch.Tensor:
        # waveform: 1-D numpy array (float32) or list; processor expects list of arrays
        # Prefer processor (which may include tokenization) but fall back to
        # feature extractor if necessary.
        if self.processor is not None:
            inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
            model_inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        elif getattr(self, "feature_extractor", None) is not None:
            inputs = self.feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
            # feature_extractor returns dict with 'input_values'
            model_inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            raise RuntimeError("No processor or feature extractor available for Wav2Vec2 model")

        with torch.no_grad():
            out = self.model(**model_inputs)
            logits = out.logits.detach().cpu()
        return logits


class MFCCCNN(nn.Module):
    """Small CNN classifier operating on mel-spectrogram images.

    The `predict_logits` helper computes a mel-spectrogram from a waveform
    and returns logits for the two classes.
    """

    def __init__(self, n_mels: int = 128, num_classes: int = 2):
        super().__init__()
        self.n_mels = n_mels
        # Simple conv stack
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        return self.net(x)

    def predict_logits(self, waveform: np.ndarray, sr: int = 16000) -> torch.Tensor:
        mel = audio_utils.compute_mel_spectrogram(waveform, sr=sr, n_mels=self.n_mels)
        # mel shape: (n_mels, T) -> convert to (1, 1, n_mels, T)
        if isinstance(mel, np.ndarray):
            mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float()
        else:
            mel_t = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.forward(mel_t).detach().cpu()
        return logits
