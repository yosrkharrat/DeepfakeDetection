"""Training loop for FFT-stream deepfake detector."""

from __future__ import annotations

import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split

from src.data.augmentation import get_eval_augmentation, get_train_augmentation
from src.data.dataset import DeepfakeDataset
from src.models.fft_stream import FFTOnlyClassifier
from src.training.losses import get_loss
from src.utils.metrics import compute_metrics, roc_auc_score_binary

# Hyperparameters (override with environment variables on Kaggle)
CSV_PATH = os.environ.get("CSV_PATH", "data/splits/train.csv")
ROOT_DIR = os.environ.get("ROOT_DIR", ".")
EPOCHS = int(os.environ.get("EPOCHS", 15))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
LR = float(os.environ.get("LR", 1e-3))
VAL_RATIO = float(os.environ.get("VAL_RATIO", 0.15))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))
SAVE_PATH = os.environ.get("SAVE_PATH", "results/checkpoints/fft_model.pth")
SEED = int(os.environ.get("SEED", 42))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))
CALIBRATE_THRESHOLD = os.environ.get("CALIBRATE_THRESHOLD", "1") not in {"0", "false", "False"}


def _set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, scheduler, device, training: bool):
	model.train() if training else model.eval()
	total_loss, all_labels, all_preds, all_probs = 0.0, [], [], []

	with torch.set_grad_enabled(training):
		for batch in loader:
			# Dataset returns (rgb, label) when use_fft=False.
			rgb, labels = batch[0].to(device), batch[-1].to(device)

			if training:
				optimizer.zero_grad()

			logits = model(rgb)
			loss = criterion(logits, labels)

			if training:
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
				optimizer.step()
				scheduler.step()

			probs = torch.softmax(logits, dim=1)[:, 1]
			total_loss += loss.item()
			all_preds += logits.argmax(1).detach().cpu().tolist()
			all_labels += labels.detach().cpu().tolist()
			all_probs += probs.detach().cpu().tolist()

	metrics = compute_metrics(all_labels, all_preds, all_probs)
	metrics["loss"] = total_loss / max(len(loader), 1)
	return metrics


def calibrate_threshold(labels: list[int], probs: list[float]) -> tuple[float, dict[str, float | int | None]]:
	labels_array = np.asarray(labels, dtype=np.int64)
	probs_array = np.asarray(probs, dtype=np.float64)
	best_threshold = 0.5
	best_metrics: dict[str, float | int | None] = {}
	best_f1 = -1.0

	for threshold in np.arange(0.1, 0.91, 0.01):
		preds = (probs_array >= threshold).astype(np.int64)
		metrics = compute_metrics(labels_array.tolist(), preds.tolist(), probs_array.tolist())
		if metrics["f1"] > best_f1:
			best_f1 = float(metrics["f1"])
			best_threshold = float(threshold)
			best_metrics = metrics

	return best_threshold, best_metrics


def train() -> None:
	_set_seed(SEED)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	# Build dataset from manifest CSV.
	full_ds = DeepfakeDataset(
		csv_path=CSV_PATH,
		root_dir=ROOT_DIR,
		transform=get_train_augmentation(),
		use_fft=False,  # FFT computed inside model.forward
	)

	val_size = int(VAL_RATIO * len(full_ds))
	train_size = len(full_ds) - val_size
	if train_size <= 0 or val_size <= 0:
		raise ValueError(
			f"Invalid split sizes for dataset length {len(full_ds)} and VAL_RATIO={VAL_RATIO}."
		)

	train_ds, val_ds = random_split(
		full_ds,
		[train_size, val_size],
		generator=torch.Generator().manual_seed(SEED),
	)

	# Keep a separate deterministic transform for validation.
	val_base_ds = DeepfakeDataset(
		csv_path=CSV_PATH,
		root_dir=ROOT_DIR,
		transform=get_eval_augmentation(),
		use_fft=False,
	)
	val_ds.dataset = val_base_ds

	pin_memory = torch.cuda.is_available()
	train_loader = DataLoader(
		train_ds,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=NUM_WORKERS,
		pin_memory=pin_memory,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=pin_memory,
	)

	labels_all = [int(rec["label"]) for rec in full_ds.records]
	real_count = labels_all.count(0)
	fake_count = labels_all.count(1)
	print(f"Class counts - real: {real_count}, fake: {fake_count}")

	model = FFTOnlyClassifier(dropout=0.3).to(device)
	criterion = get_loss(real_count=real_count, fake_count=fake_count).to(device)
	optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.OneCycleLR(
		optimizer,
		max_lr=LR,
		steps_per_epoch=max(len(train_loader), 1),
		epochs=EPOCHS,
		pct_start=0.1,
		div_factor=10,
		final_div_factor=100,
	)

	best_auc = 0.0
	save_path = Path(SAVE_PATH)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	for epoch in range(1, EPOCHS + 1):
		train_m = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, training=True)
		val_m = run_epoch(model, val_loader, criterion, optimizer, scheduler, device, training=False)

		print(
			f"Epoch {epoch:02d}/{EPOCHS} | "
			f"Train loss {train_m['loss']:.4f} acc {train_m['accuracy']:.3f} | "
			f"Val loss {val_m['loss']:.4f} acc {val_m['accuracy']:.3f} "
			f"F1 {val_m['f1']:.3f} AUC {val_m['auc_roc']:.3f}"
		)

		if val_m["auc_roc"] > best_auc:
			best_auc = val_m["auc_roc"]
			torch.save(model.state_dict(), save_path)
			print(f"  Saved best model (AUC {best_auc:.3f}) -> {save_path}")

	print(f"\nTraining complete. Best val AUC: {best_auc:.3f}")

	if CALIBRATE_THRESHOLD:
		model.load_state_dict(torch.load(save_path, map_location=device))
		model.eval()
		all_labels, all_probs = [], []
		with torch.no_grad():
			for batch in val_loader:
				rgb, labels = batch[0].to(device), batch[-1].to(device)
				logits = model(rgb)
				probs = torch.softmax(logits, dim=1)[:, 1]
				all_labels.extend(labels.detach().cpu().tolist())
				all_probs.extend(probs.detach().cpu().tolist())

		best_threshold, best_metrics = calibrate_threshold(all_labels, all_probs)
		cm = confusion_matrix(all_labels, [1 if p >= best_threshold else 0 for p in all_probs])
		metrics_path = save_path.with_suffix(".metrics.json")
		metrics_path.write_text(
			json.dumps(
				{
					"best_threshold": best_threshold,
					"best_val_auc": best_auc,
					"calibrated_metrics": best_metrics,
					"confusion_matrix": cm.tolist(),
				},
				indent=2,
			),
			encoding="utf-8",
		)
		print(f"Saved calibrated threshold + metrics to {metrics_path}")


if __name__ == "__main__":
	train()
