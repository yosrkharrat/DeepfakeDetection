"""Face detector wrapper with MTCNN primary backend and OpenCV fallback."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
	import cv2
except ImportError:  # pragma: no cover - runtime env dependent
	cv2 = None


class FaceDetector:
	"""Detect faces and return bounding boxes/crops.

	Backends:
	- MTCNN from facenet-pytorch when available.
	- OpenCV Haar cascade fallback.
	"""

	def __init__(self, device: str = "cpu") -> None:
		if cv2 is None:
			raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")

		self.backend = "none"
		self.mtcnn = None
		self.haar = None

		try:
			from facenet_pytorch import MTCNN  # type: ignore

			self.mtcnn = MTCNN(keep_all=True, device=device)
			self.backend = "mtcnn"
			return
		except Exception:
			pass

		cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
		if cascade_path.exists():
			self.haar = cv2.CascadeClassifier(str(cascade_path))
			self.backend = "haar"

		if self.backend == "none":
			raise RuntimeError(
				"No face detection backend found. Install facenet-pytorch or ensure OpenCV haarcascades exist."
			)

	def detect_boxes(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
		"""Return face boxes as (x, y, w, h)."""
		if self.backend == "mtcnn" and self.mtcnn is not None:
			image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
			boxes, _ = self.mtcnn.detect(image_rgb)
			if boxes is None:
				return []
			h, w = image_bgr.shape[:2]
			out: List[Tuple[int, int, int, int]] = []
			for box in boxes:
				x1, y1, x2, y2 = box.tolist()
				x1_i = max(0, int(x1))
				y1_i = max(0, int(y1))
				x2_i = min(w, int(x2))
				y2_i = min(h, int(y2))
				width = x2_i - x1_i
				height = y2_i - y1_i
				if width > 0 and height > 0:
					out.append((x1_i, y1_i, width, height))
			return out

		if self.backend == "haar" and self.haar is not None:
			gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
			boxes = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
			return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]

		return []

	def detect_and_crop(
		self,
		image_bgr: np.ndarray,
		crop_size: int = 224,
		min_face_size: int = 64,
	) -> List[np.ndarray]:
		"""Detect faces and return resized BGR crops."""
		if cv2 is None:
			raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")

		crops: List[np.ndarray] = []
		for x, y, w, h in self.detect_boxes(image_bgr):
			if w < min_face_size or h < min_face_size:
				continue
			crop = image_bgr[y : y + h, x : x + w]
			if crop.size == 0:
				continue
			crops.append(cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA))
		return crops

	def detect_from_path(
		self,
		image_path: Union[str, Path],
		crop_size: int = 224,
		min_face_size: int = 64,
	) -> List[np.ndarray]:
		"""Load an image from disk and return face crops."""
		if cv2 is None:
			raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")

		img = cv2.imread(str(image_path))
		if img is None:
			return []
		return self.detect_and_crop(img, crop_size=crop_size, min_face_size=min_face_size)

