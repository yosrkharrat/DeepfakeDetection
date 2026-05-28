"""Loss helpers for training deepfake classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn


def get_loss(real_count: int | None = None, fake_count: int | None = None) -> nn.Module:
	"""Return cross-entropy, optionally class-weighted for imbalanced data.

	When both class counts are provided, weights are computed as inverse-frequency
	style balancing to upweight the minority class.
	"""
	if real_count and fake_count:
		total = real_count + fake_count
		weight = torch.tensor(
			[
				total / (2 * real_count),  # class 0: real
				total / (2 * fake_count),  # class 1: fake
			],
			dtype=torch.float32,
		)
		return nn.CrossEntropyLoss(weight=weight)

	return nn.CrossEntropyLoss()
