"""PyTorch dataset for dual-stream deepfake training.

Expected CSV columns:
- path
- label
- source_dataset
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
	from PIL import Image
except ImportError as exc:  # pragma: no cover - runtime env dependent
	raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc

from src.utils.fft_transform import compute_fft_magnitude

logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
	"""Loads face crops and returns RGB + FFT tensors with label."""

	REQUIRED_COLUMNS = {"path", "label", "source_dataset"}

	def __init__(
		self,
		csv_path: str | Path,
		root_dir: Optional[str | Path] = None,
		transform: Optional[Any] = None,
		use_fft: bool = True,
		return_metadata: bool = False,
	) -> None:
		self.csv_path = Path(csv_path)
		self.root_dir = Path(root_dir) if root_dir is not None else Path.cwd()
		self.transform = transform
		self.use_fft = use_fft
		self.return_metadata = return_metadata
		self.records = self._load_records(self.csv_path)
		self._validate_paths()

	def _load_records(self, csv_path: Path) -> List[Dict[str, str]]:
		if not csv_path.exists():
			raise FileNotFoundError(f"CSV not found: {csv_path}")

		with csv_path.open("r", newline="", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			cols = set(reader.fieldnames or [])
			missing = self.REQUIRED_COLUMNS - cols
			if missing:
				raise ValueError(f"CSV missing required columns: {sorted(missing)}")

			rows: List[Dict[str, str]] = []
			for i, row in enumerate(reader):
				if row["path"] is None or row["label"] is None:
					raise ValueError(f"Invalid row at index {i}: path/label cannot be null")
				rows.append(row)

		if not rows:
			raise ValueError(f"CSV has no rows: {csv_path}")

		return rows

	def _validate_paths(self) -> None:
		"""Warn about missing image files at init time instead of crashing mid-epoch."""
		missing = [
			rec["path"]
			for rec in self.records
			if not self._resolve_image_path(rec["path"]).exists()
		]
		if missing:
			logger.warning(
				"%d/%d image paths do not exist. First missing: %s",
				len(missing),
				len(self.records),
				missing[0],
			)

	def __len__(self) -> int:
		return len(self.records)

	def _resolve_image_path(self, value: str) -> Path:
		p = Path(value)
		return p if p.is_absolute() else (self.root_dir / p)

	@staticmethod
	def _image_to_tensor(image_hwc: np.ndarray) -> torch.Tensor:
		tensor = torch.from_numpy(image_hwc).permute(2, 0, 1).contiguous().float()
		# A.Normalize outputs float32 already in normalized range; uint8 needs /255
		if image_hwc.dtype == np.uint8:
			return tensor / 255.0
		return tensor

	def _apply_transform(self, image_hwc: np.ndarray) -> np.ndarray:
		if self.transform is None:
			return image_hwc

		transformed = self.transform(image=image_hwc)
		if isinstance(transformed, dict) and "image" in transformed:
			return transformed["image"]
		return transformed

	def __getitem__(self, idx: int):
		rec = self.records[idx]
		img_path = self._resolve_image_path(rec["path"])
		label = int(rec["label"])
		source_dataset = rec["source_dataset"]

		if not img_path.exists():
			logger.warning("Image not found, skipping: %s", img_path)
			return self.__getitem__((idx + 1) % len(self.records))

		with Image.open(img_path) as img:
			image_rgb = img.convert("RGB")
			image_np = np.array(image_rgb)

		image_np = self._apply_transform(image_np)
		rgb_tensor = self._image_to_tensor(image_np)

		fft_tensor: Optional[torch.Tensor]
		if self.use_fft:
			fft_tensor = compute_fft_magnitude(rgb_tensor.unsqueeze(0)).squeeze(0)
		else:
			fft_tensor = None

		label_tensor = torch.tensor(label, dtype=torch.long)

		if self.return_metadata:
			metadata = {
				"path": rec["path"],
				"source_dataset": source_dataset,
			}
			if fft_tensor is None:
				return rgb_tensor, label_tensor, metadata
			return rgb_tensor, fft_tensor, label_tensor, metadata

		if fft_tensor is None:
			return rgb_tensor, label_tensor
		return rgb_tensor, fft_tensor, label_tensor

