"""Claim verification model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


@dataclass
class ClaimVerifyResult:
    label: str
    confidence: float
    p_support: float
    p_refute: float
    p_nei: float


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _find_label_index(labels: dict[int, str], keys: Iterable[str]) -> int | None:
    for idx, label in labels.items():
        norm = _normalize_label(label)
        if any(key in norm for key in keys):
            return int(idx)
    return None


class ClaimVerifyDetector:
    """Wrapper around a claim verification model (supports NLI-style checkpoints)."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        use_hf_inference: bool = False,
        provider: str = "hf-inference",
        api_key: str | None = None,
        support_label: str | None = None,
        refute_label: str | None = None,
        neutral_label: str | None = None,
    ) -> None:
        self.device = device
        self.use_hf_inference = bool(use_hf_inference)
        self.model_id = checkpoint_path
        self.support_label = support_label
        self.refute_label = refute_label
        self.neutral_label = neutral_label

        if self.use_hf_inference:
            if InferenceClient is None:
                raise RuntimeError("huggingface_hub is required for HF inference. Install huggingface_hub.")
            self.client = InferenceClient(provider=provider, api_key=api_key)
            self.tokenizer = None
            self.model = None
            self.id2label = {}
            self.num_labels = 3
            self.entailment_idx, self.contradiction_idx, self.neutral_idx = (None, None, None)
        else:
            self.client = None
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
            self.model.to(self.device)
            self.model.eval()

            id2label = getattr(self.model.config, "id2label", None) or {}
            self.id2label = {int(k): str(v) for k, v in id2label.items()} if id2label else {}
            self.num_labels = int(getattr(self.model.config, "num_labels", len(self.id2label) or 2))

            self.entailment_idx, self.contradiction_idx, self.neutral_idx = self._resolve_label_indices()

    def _resolve_label_indices(self) -> tuple[int | None, int | None, int | None]:
        if self.id2label:
            entailment = _find_label_index(self.id2label, ("entail", "support", "true", "yes"))
            contradiction = _find_label_index(self.id2label, ("contradict", "refute", "false", "no"))
            neutral = _find_label_index(self.id2label, ("neutral", "unknown", "uncertain", "not_enough"))
        else:
            entailment = contradiction = neutral = None

        if self.num_labels == 3 and (entailment is None or contradiction is None or neutral is None):
            # Default MNLI ordering: contradiction, neutral, entailment
            contradiction = contradiction if contradiction is not None else 0
            neutral = neutral if neutral is not None else 1
            entailment = entailment if entailment is not None else 2

        if self.num_labels == 2 and entailment is None and contradiction is None:
            # Assume label 1 is positive/support and label 0 is negative/refute
            contradiction = 0
            entailment = 1
            neutral = None

        return entailment, contradiction, neutral

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
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        total = sum(scores.values())
        if total <= 0.0:
            return scores
        return {key: value / total for key, value in scores.items()}

    def _parse_remote_scores(self, results: list[dict[str, object]]) -> tuple[float, float, float]:
        p_support = None
        p_refute = None
        p_nei = None

        if self.support_label:
            p_support = self._score_for_label(results, self.support_label)
        if self.refute_label:
            p_refute = self._score_for_label(results, self.refute_label)
        if self.neutral_label:
            p_nei = self._score_for_label(results, self.neutral_label)

        if p_support is None:
            p_support = self._score_for_keywords(results, ("support", "entail", "true", "yes"))
        if p_refute is None:
            p_refute = self._score_for_keywords(results, ("refute", "contradict", "false", "no"))
        if p_nei is None:
            p_nei = self._score_for_keywords(results, ("neutral", "unknown", "not_enough", "uncertain"))

        if p_support is None and p_refute is None and p_nei is None:
            top = max(results, key=lambda item: float(item.get("score", 0.0)))
            p_support = float(top.get("score", 0.0))
            p_refute = 0.0
            p_nei = 0.0

        scores = {
            "support": float(p_support or 0.0),
            "refute": float(p_refute or 0.0),
            "nei": float(p_nei or 0.0),
        }
        scores = self._normalize_scores(scores)
        return scores["support"], scores["refute"], scores["nei"]

    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def _predict_with_evidence(self, claim: str, evidence: str) -> ClaimVerifyResult:
        if self.use_hf_inference:
            if self.client is None:
                raise RuntimeError("HF inference client not initialized.")
            prompt = f"Evidence: {evidence}\nClaim: {claim}"
            result = self.client.text_classification(prompt, model=self.model_id)
            results = result if isinstance(result, list) else [result]
            p_support, p_refute, p_nei = self._parse_remote_scores(results)
            label = "SUPPORTED"
            confidence = p_support
            if p_refute > confidence:
                label = "REFUTED"
                confidence = p_refute
            if p_nei > confidence:
                label = "NOT_ENOUGH_INFO"
                confidence = p_nei
            return ClaimVerifyResult(
                label=label,
                confidence=round(confidence, 4),
                p_support=round(p_support, 4),
                p_refute=round(p_refute, 4),
                p_nei=round(p_nei, 4),
            )

        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = self._softmax(logits)

        p_support = float(probs[self.entailment_idx]) if self.entailment_idx is not None else 0.0
        p_refute = float(probs[self.contradiction_idx]) if self.contradiction_idx is not None else 0.0
        p_nei = float(probs[self.neutral_idx]) if self.neutral_idx is not None else 0.0

        label = "SUPPORTED"
        confidence = p_support
        if p_refute > confidence:
            label = "REFUTED"
            confidence = p_refute
        if p_nei > confidence:
            label = "NOT_ENOUGH_INFO"
            confidence = p_nei

        return ClaimVerifyResult(
            label=label,
            confidence=round(confidence, 4),
            p_support=round(p_support, 4),
            p_refute=round(p_refute, 4),
            p_nei=round(p_nei, 4),
        )

    def _predict_claim_only(self, claim: str) -> ClaimVerifyResult:
        if self.use_hf_inference:
            if self.client is None:
                raise RuntimeError("HF inference client not initialized.")
            result = self.client.text_classification(claim, model=self.model_id)
            results = result if isinstance(result, list) else [result]
            p_support, p_refute, p_nei = self._parse_remote_scores(results)
            label = "SUPPORTED"
            confidence = p_support
            if p_refute > confidence:
                label = "REFUTED"
                confidence = p_refute
            if p_nei > confidence:
                label = "NOT_ENOUGH_INFO"
                confidence = p_nei
            return ClaimVerifyResult(
                label=label,
                confidence=round(confidence, 4),
                p_support=round(p_support, 4),
                p_refute=round(p_refute, 4),
                p_nei=round(p_nei, 4),
            )

        # If we have a binary classifier, use single-text prediction directly.
        if self.num_labels == 2:
            inputs = self.tokenizer(
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits[0]
            probs = self._softmax(logits)

            p_refute = float(probs[self.contradiction_idx]) if self.contradiction_idx is not None else 0.0
            p_support = float(probs[self.entailment_idx]) if self.entailment_idx is not None else 0.0
            label = "SUPPORTED" if p_support >= p_refute else "REFUTED"
            confidence = max(p_support, p_refute)
            return ClaimVerifyResult(
                label=label,
                confidence=round(confidence, 4),
                p_support=round(p_support, 4),
                p_refute=round(p_refute, 4),
                p_nei=0.0,
            )

        # Zero-shot style using NLI entailment scores for claim-only input.
        hypotheses = [
            ("SUPPORTED", "This statement is true."),
            ("REFUTED", "This statement is false."),
            ("NOT_ENOUGH_INFO", "This statement is uncertain."),
        ]
        scores = []
        for _, hypothesis in hypotheses:
            inputs = self.tokenizer(
                claim,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits[0]

            entailment_score = logits[self.entailment_idx] if self.entailment_idx is not None else logits.max()
            scores.append(entailment_score)

        scores_t = torch.stack(scores)
        probs = self._softmax(scores_t)

        p_support = float(probs[0].item())
        p_refute = float(probs[1].item())
        p_nei = float(probs[2].item())

        label = "SUPPORTED"
        confidence = p_support
        if p_refute > confidence:
            label = "REFUTED"
            confidence = p_refute
        if p_nei > confidence:
            label = "NOT_ENOUGH_INFO"
            confidence = p_nei

        return ClaimVerifyResult(
            label=label,
            confidence=round(confidence, 4),
            p_support=round(p_support, 4),
            p_refute=round(p_refute, 4),
            p_nei=round(p_nei, 4),
        )

    def predict(self, claim: str, evidence: str | None = None) -> dict[str, object]:
        """Return a prediction dict for a claim, optionally with evidence."""
        if not isinstance(claim, str):
            raise TypeError("claim must be a string")
        evidence_text = evidence.strip() if isinstance(evidence, str) and evidence.strip() else None

        if evidence_text:
            result = self._predict_with_evidence(claim.strip(), evidence_text)
        else:
            result = self._predict_claim_only(claim.strip())

        return {
            "label": result.label,
            "confidence": result.confidence,
            "p_support": result.p_support,
            "p_refute": result.p_refute,
            "p_nei": result.p_nei,
        }
