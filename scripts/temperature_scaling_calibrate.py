#!/usr/bin/env python3
"""Fit a single temperature scalar for probability calibration.

Usage: python scripts/temperature_scaling_calibrate.py
Environment:
  CHECKPOINT (optional) - path to checkpoint (used by service.from_checkpoint)
  MODE (optional) - 'ensemble' recommended

The script loads a small calibration set from
`data/processed/processed/precompute_manifest.csv` and fits a single
temperature T (T>0) minimizing negative log-likelihood on the predicted
positive-class probabilities.
"""
from pathlib import Path
import json
import os
import math
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.utils.inference import FusionInferenceService


MANIFEST = Path("data/processed/processed/precompute_manifest.csv")
OUT_PATH = Path("results/checkpoints/temperature.json")
SAMPLE_MAX = 400


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    records = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            path = parts[0]
            try:
                label = int(parts[1])
            except Exception:
                continue
            records.append((path, label))
    return records


def main():
    ckpt = os.environ.get("CHECKPOINT", "results/checkpoints/fusion_best.pt")
    mode = os.environ.get("MODE", "ensemble")
    device = os.environ.get("DEVICE", "cpu")

    try:
        service = FusionInferenceService.from_checkpoint(ckpt, device=device, mode=mode)
    except Exception as exc:
        print(f"Unable to load in mode={mode}: {exc}\nFalling back to mode='fusion'.")
        service = FusionInferenceService.from_checkpoint(ckpt, device=device, mode="fusion")

    records = load_manifest(MANIFEST)
    if not records:
        print("No manifest records found; aborting.")
        return

    # Sample subset for calibration
    sample = records[: min(len(records), SAMPLE_MAX)]
    logits = []
    labels = []

    print(f"Collecting predictions for {len(sample)} calibration images...")
    for path, label in sample:
        p = None
        try:
            res = service.predict_path(path)
            # take best face fake probability
            p = float(res["confidence"]) if res is not None else None
        except Exception as exc:
            # skip unreadable/missing images
            print(f"Skipping {path}: {exc}")
            continue
        if p is None:
            continue
        # Clip probabilities to avoid extreme log-odds
        eps = 1e-6
        p = min(max(p, eps), 1.0 - eps)
        logit = math.log(p / (1.0 - p))
        logits.append(logit)
        labels.append(label)

    if not logits:
        print("No valid predictions collected; aborting.")
        return

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    # Optimize logT where T = exp(logT) to ensure positivity.
    logT = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([logT], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        T = torch.exp(logT)
        scaled = logits_t / T
        # p = sigmoid(scaled) (positive-class probability)
        p = torch.sigmoid(scaled)
        # Negative log-likelihood
        loss = - (labels_t * torch.log(p + 1e-12) + (1 - labels_t) * torch.log(1 - p + 1e-12)).mean()
        loss.backward()
        return loss

    print("Fitting temperature (this may take a moment)...")
    try:
        optimizer.step(closure)
    except Exception:
        # LBFGS can be finicky in some envs — fallback to simple Adam
        logT = torch.tensor(0.0, requires_grad=True)
        optimizer = torch.optim.Adam([logT], lr=0.1)
        for _ in range(200):
            optimizer.zero_grad()
            T = torch.exp(logT)
            scaled = logits_t / T
            p = torch.sigmoid(scaled)
            loss = - (labels_t * torch.log(p + 1e-12) + (1 - labels_t) * torch.log(1 - p + 1e-12)).mean()
            loss.backward()
            optimizer.step()

    T_final = float(torch.exp(logT).detach().cpu().item())
    print(f"Fitted temperature: {T_final:.6f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"temperature": T_final}, fh)

    print(f"Saved temperature to {OUT_PATH}")


if __name__ == "__main__":
    main()
