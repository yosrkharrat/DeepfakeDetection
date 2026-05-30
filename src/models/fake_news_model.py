"""Fake news detection wrapper for text inputs."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


class FakeNewsDetector:
    """Wrapper around a text classifier checkpoint (local or HF inference)."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        use_hf_inference: bool = False,
        provider: str = "hf-inference",
        api_key: str | None = None,
        positive_label: str | None = None,
        negative_label: str | None = None,
    ) -> None:
        self.device = device
        self.use_hf_inference = bool(use_hf_inference)
        self.model_id = checkpoint_path
        self.positive_label = positive_label
        self.negative_label = negative_label

        if self.use_hf_inference:
            if InferenceClient is None:
                raise RuntimeError("huggingface_hub is required for HF inference. Install huggingface_hub.")
            self.client = InferenceClient(provider=provider, api_key=api_key)
            self.tokenizer = None
            self.model = None
        else:
            self.client = None
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            self.fake_idx, self.real_idx = self._resolve_label_indices()

    def _resolve_label_indices(self) -> tuple[int, int]:
        """Return (fake_idx, real_idx) by inspecting model.config.id2label."""
        id2label = getattr(self.model.config, "id2label", None) or {}
        id2label = {int(k): str(v) for k, v in id2label.items()}
        fake_idx = None
        real_idx = None
        for idx, label in id2label.items():
            norm = self._normalize_label(label)
            if any(k in norm for k in ("fake", "label_1", "1", "positive", "yes", "true")):
                fake_idx = idx
            elif any(k in norm for k in ("real", "label_0", "0", "negative", "no", "false")):
                real_idx = idx
        # Default fallback: 0 = real, 1 = fake (standard convention)
        if fake_idx is None:
            fake_idx = 1
        if real_idx is None:
            real_idx = 0
        return fake_idx, real_idx

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label.strip().lower().replace(" ", "_")

    def _score_for_label(self, results: list[dict[str, object]], label: str) -> float | None:
        target = self._normalize_label(label)
        for item in results:
            raw_label = str(item.get("label", ""))
            if self._normalize_label(raw_label) == target:
                return float(item.get("score", 0.0))
        return None

    def _score_for_keywords(self, results: list[dict[str, object]], keywords: tuple[str, ...]) -> float | None:
        for item in results:
            raw_label = str(item.get("label", ""))
            norm = self._normalize_label(raw_label)
            if any(key in norm for key in keywords):
                return float(item.get("score", 0.0))
        return None

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _parse_remote_scores(self, results: list[dict[str, object]]) -> tuple[float, float]:
        p_fake = None
        p_real = None

        if self.positive_label:
            p_fake = self._score_for_label(results, self.positive_label)
        if self.negative_label:
            p_real = self._score_for_label(results, self.negative_label)

        if p_fake is None:
            p_fake = self._score_for_keywords(results, ("fake", "label_1", "1", "positive", "yes", "true"))
        if p_real is None:
            p_real = self._score_for_keywords(results, ("real", "label_0", "0", "negative", "no", "false"))

        if p_fake is None and p_real is None:
            # Fallback: treat top label as fake when no mapping is available.
            top = max(results, key=lambda item: float(item.get("score", 0.0)))
            p_fake = float(top.get("score", 0.0))
            p_real = 1.0 - p_fake
        elif p_fake is None and p_real is not None:
            p_fake = 1.0 - p_real
        elif p_real is None and p_fake is not None:
            p_real = 1.0 - p_fake

        return self._clamp01(float(p_fake)), self._clamp01(float(p_real))

    def predict(self, text: str) -> dict[str, object]:
        """Return a prediction dict for a single news article or headline."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if self.use_hf_inference:
            if self.client is None:
                raise RuntimeError("HF inference client not initialized.")
            result = self.client.text_classification(text, model=self.model_id)
            results = result if isinstance(result, list) else [result]
            p_fake, p_real = self._parse_remote_scores(results)
        else:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]

            p_fake = float(probs[self.fake_idx].item())
            p_real = float(probs[self.real_idx].item())
        is_fake = p_fake >= 0.5
        confidence = p_fake if is_fake else p_real

        return {
            "is_fake": bool(is_fake),
            "confidence": round(confidence, 4),
            "p_fake": round(p_fake, 4),
            "p_real": round(p_real, 4),
            "label": "FAKE" if is_fake else "REAL",
        }
