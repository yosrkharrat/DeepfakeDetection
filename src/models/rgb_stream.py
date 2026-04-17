"""
RGB Stream — Deepfake Detection System
ResNet-18 backbone that extracts semantic features from face crops.

How it works:
  1. Start from a ResNet-18 image backbone
  2. Remove the default 1000-class classifier
  3. Replace it with a projection head that outputs a 512-d vector
  4. Feed this vector to the dual-stream fusion module
"""

from __future__ import annotations

import importlib
import torch
import torch.nn as nn


class RGBStreamResNet(nn.Module):
	"""
	ResNet-18 RGB backbone that outputs a 512-d feature vector.

	Expected input:
		(B, 3, H, W), typically H=W=224 after preprocessing.

	Output:
		(B, 512) feature embedding used by the fusion model.
	"""

	def __init__(self, pretrained: bool = True, dropout: float = 0.2):
		super().__init__()

		try:
			tv_models = importlib.import_module("torchvision.models")
		except ImportError as exc:
			raise ImportError(
				"torchvision is required for RGBStreamResNet. Install it with: pip install torchvision"
			) from exc

		resnet18 = getattr(tv_models, "resnet18")
		weights_enum = getattr(tv_models, "ResNet18_Weights", None)

		# torchvision >= 0.13 uses the weights API.
		if weights_enum is not None:
			weights = weights_enum.DEFAULT if pretrained else None
			backbone = resnet18(weights=weights)
		else:
			# torchvision < 0.13 fallback.
			backbone = resnet18(pretrained=pretrained)

		in_features = backbone.fc.in_features
		backbone.fc = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(in_features, 512),
		)
		self.backbone = backbone

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return RGB feature embeddings of shape (B, 512)."""
		return self.backbone(x)


class RGBOnlyClassifier(nn.Module):
	"""
	Optional RGB-only baseline classifier for ablations.
	"""

	def __init__(self, pretrained: bool = True, dropout: float = 0.2):
		super().__init__()
		self.backbone = RGBStreamResNet(pretrained=pretrained, dropout=dropout)
		self.classifier = nn.Sequential(
			nn.Linear(512, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout),
			nn.Linear(128, 2),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone(x)
		return self.classifier(features)


if __name__ == "__main__":
	print("=== RGB Stream sanity check ===\n")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	dummy_rgb = torch.rand(4, 3, 224, 224).to(device)

	model = RGBStreamResNet(pretrained=False).to(device)
	features = model(dummy_rgb)
	print(f"Feature vector shape: {features.shape}")  # (4, 512)

	clf = RGBOnlyClassifier(pretrained=False).to(device)
	logits = clf(dummy_rgb)
	print(f"Classifier logits shape: {logits.shape}")  # (4, 2)

	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"\nRGBStreamResNet trainable params: {n_params:,}")

	print("\nAll checks passed.")
