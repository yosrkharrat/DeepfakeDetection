"""Generate stratified train/val/test CSV manifests from processed face crops.

CSV schema (contract):
- path
- label (0 real, 1 fake)
- source_dataset
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CLASS_LABEL_MAP = {
	"real": 0,
	"original": 0,
	"fake": 1,
	"deepfakes": 1,
	"face2face": 1,
	"faceswap": 1,
	"neuraltextures": 1,
	"faceshifter": 1,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Create stratified train/val/test splits from processed images.")
	parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Processed image root.")
	parser.add_argument("--output-dir", type=Path, default=Path("data/splits"), help="Where split CSV files are saved.")
	parser.add_argument("--train-ratio", type=float, default=0.70, help="Training split ratio.")
	parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
	parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffling.")
	parser.add_argument(
		"--strict-dataset-layout",
		action="store_true",
		help="Require layout like data/processed/<dataset>/<class>/...",
	)
	return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
	total = train_ratio + val_ratio + test_ratio
	if abs(total - 1.0) > 1e-9:
		raise ValueError(f"Ratios must sum to 1.0. Got {total:.6f}.")


def infer_record(image_path: Path, processed_root: Path, strict_layout: bool) -> Optional[Dict[str, object]]:
	rel = image_path.relative_to(processed_root)
	parts = [p.lower() for p in rel.parts]

	if strict_layout and len(rel.parts) < 2:
		return None

	source_dataset = rel.parts[0] if len(rel.parts) >= 2 else "unknown"
	label = None

	for part in parts:
		if part in CLASS_LABEL_MAP:
			label = CLASS_LABEL_MAP[part]
			break

	if label is None:
		return None

	try:
		rel_to_workspace = image_path.relative_to(Path.cwd())
		path_value = rel_to_workspace.as_posix()
	except ValueError:
		path_value = image_path.as_posix()

	return {
		"path": path_value,
		"label": label,
		"source_dataset": source_dataset,
	}


def discover_records(processed_root: Path, strict_layout: bool) -> List[Dict[str, object]]:
	records: List[Dict[str, object]] = []
	for image_path in processed_root.rglob("*"):
		if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
			continue
		rec = infer_record(image_path, processed_root, strict_layout=strict_layout)
		if rec is not None:
			records.append(rec)
	return records


def stratified_split(
	records: List[Dict[str, object]],
	train_ratio: float,
	val_ratio: float,
	seed: int,
) -> Dict[str, List[Dict[str, object]]]:
	by_label: Dict[int, List[Dict[str, object]]] = defaultdict(list)
	for rec in records:
		by_label[int(rec["label"])].append(rec)

	rng = random.Random(seed)
	train, val, test = [], [], []

	for _, items in by_label.items():
		rng.shuffle(items)
		n = len(items)
		n_train = int(n * train_ratio)
		n_val = int(n * val_ratio)

		label_train = items[:n_train]
		label_val = items[n_train : n_train + n_val]
		label_test = items[n_train + n_val :]

		train.extend(label_train)
		val.extend(label_val)
		test.extend(label_test)

	rng.shuffle(train)
	rng.shuffle(val)
	rng.shuffle(test)

	return {"train": train, "val": val, "test": test}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["path", "label", "source_dataset"])
		writer.writeheader()
		writer.writerows(rows)


def print_summary(name: str, rows: List[Dict[str, object]]) -> None:
	label_counts = Counter(int(r["label"]) for r in rows)
	print(
		f"{name:<5} | total={len(rows):>6} | real(0)={label_counts.get(0, 0):>6} | fake(1)={label_counts.get(1, 0):>6}"
	)


def main() -> int:
	args = parse_args()

	try:
		validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
	except ValueError as exc:
		print(f"[ERROR] {exc}")
		return 1

	if not args.processed_dir.exists():
		print(f"[ERROR] Processed directory not found: {args.processed_dir}")
		return 2

	records = discover_records(args.processed_dir, strict_layout=args.strict_dataset_layout)
	if not records:
		print("[ERROR] No valid image records found under processed directory.")
		print("        Expected image files with class names in path (real/original or fake/deepfakes/...).")
		return 3

	splits = stratified_split(
		records,
		train_ratio=args.train_ratio,
		val_ratio=args.val_ratio,
		seed=args.seed,
	)

	train_csv = args.output_dir / "train.csv"
	val_csv = args.output_dir / "val.csv"
	test_csv = args.output_dir / "test.csv"

	write_csv(train_csv, splits["train"])
	write_csv(val_csv, splits["val"])
	write_csv(test_csv, splits["test"])

	print("\n=== Split Summary ===")
	print_summary("all", records)
	print_summary("train", splits["train"])
	print_summary("val", splits["val"])
	print_summary("test", splits["test"])
	print(f"\nWrote: {train_csv}")
	print(f"Wrote: {val_csv}")
	print(f"Wrote: {test_csv}")

	return 0


if __name__ == "__main__":
	sys.exit(main())
