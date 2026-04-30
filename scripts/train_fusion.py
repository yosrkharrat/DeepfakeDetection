"""Train the dual-stream fusion classifier from RGB inputs and pretrained backbones."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion_model import FusionModel
from src.data.augmentation import get_eval_augmentation, get_train_augmentation
from src.data.dataset import DeepfakeDataset
from src.models.fft_stream import FFTStreamCNN, normalize_fft_checkpoint_state_dict
from src.models.rgb_stream import RGBStreamResNet, normalize_rgb_checkpoint_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RGB+FFT fusion model.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--root-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--rgb-checkpoint", type=Path, default=None)
    parser.add_argument("--fft-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--fft-preprocessing",
        choices=("auto", "fft2", "rfft2"),
        default="auto",
        help=(
            "FFT preprocessing variant for the FFT backbone. "
            "'auto' detects Kaggle FFTOnlyClassifier checkpoints and switches to rfft2."
        ),
    )
    parser.add_argument(
        "--freeze-backbones",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze RGB/FFT backbones and train only the fusion classifier.",
    )
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require checkpoint keys to match exactly when loading backbones.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must deserialize into a mapping.")
    return config


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_arg(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not all(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_serialized_checkpoint(checkpoint_path: Path) -> Any:
    """Load checkpoints saved either as plain weights or full training dictionaries."""
    try:
        return torch.load(checkpoint_path, map_location="cpu")
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    checkpoint = load_serialized_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}.")

    state_dict = strip_module_prefix(checkpoint)
    if isinstance(model, RGBStreamResNet):
        state_dict = normalize_rgb_checkpoint_state_dict(state_dict)
    elif isinstance(model, FFTStreamCNN):
        state_dict, inferred_rfft = normalize_fft_checkpoint_state_dict(state_dict)
        if inferred_rfft:
            model.set_use_rfft(True)
        state_dict["_use_rfft_flag"] = torch.tensor(model.uses_rfft, dtype=torch.bool)
    incompatible = model.load_state_dict(state_dict, strict=strict)

    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            f"Loaded {checkpoint_path} with missing keys={incompatible.missing_keys} "
            f"and unexpected keys={incompatible.unexpected_keys}"
        )


def build_dataloader(
    csv_path: Path,
    root_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    transform = get_train_augmentation(image_size) if train else get_eval_augmentation(image_size)
    dataset = DeepfakeDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        transform=transform,
        use_fft=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def compute_class_weights(dataset: DeepfakeDataset) -> torch.Tensor:
    labels = torch.tensor([int(record["label"]) for record in dataset.records], dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=2).float().clamp_min(1.0)
    weights = labels.numel() / (class_counts.numel() * class_counts)
    return weights


def train_one_epoch(
    model: FusionModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for batch in progress:
        if len(batch) == 3:
            rgb_batch, _, labels = batch
        else:
            rgb_batch, labels = batch
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(rgb_batch, rgb_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def evaluate(
    model: FusionModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(dataloader, desc="Val", leave=False)
    for batch in progress:
        if len(batch) == 3:
            rgb_batch, _, labels = batch
        else:
            rgb_batch, labels = batch
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(rgb_batch, rgb_batch)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def save_checkpoint(
    output_path: Path,
    model: FusionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    image_size = int(get_nested(config, "data", "image_size", default=224))
    splits_dir = resolve_project_path(get_nested(config, "data", "splits_dir", default="data/splits"))
    output_dir = resolve_project_path(
        resolve_arg(args.output_dir, get_nested(config, "paths", "checkpoints", default="results/checkpoints"))
    )
    batch_size = int(resolve_arg(args.batch_size, get_nested(config, "training", "batch_size", default=32)))
    epochs = int(resolve_arg(args.epochs, get_nested(config, "training", "epochs", default=10)))
    learning_rate = float(
        resolve_arg(args.learning_rate, get_nested(config, "training", "learning_rate", default=1e-4))
    )
    num_workers = int(resolve_arg(args.num_workers, get_nested(config, "training", "num_workers", default=0)))
    seed = int(resolve_arg(args.seed, get_nested(config, "training", "seed", default=42)))
    early_stopping_patience = int(
        resolve_arg(
            args.early_stopping_patience,
            get_nested(config, "training", "early_stopping_patience", default=5),
        )
    )
    if early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1.")
    device_name = str(
        resolve_arg(
            args.device,
            get_nested(
                config,
                "training",
                "device",
                default="cuda" if torch.cuda.is_available() else "cpu",
            ),
        )
    )
    freeze_backbones = bool(resolve_arg(args.freeze_backbones, True))
    lr_step_size = int(resolve_arg(None, get_nested(config, "training", "lr_step_size", default=10)))
    lr_gamma = float(resolve_arg(None, get_nested(config, "training", "lr_gamma", default=0.5)))

    train_csv = resolve_project_path(resolve_arg(args.train_csv, splits_dir / "train.csv"))
    val_csv = resolve_project_path(resolve_arg(args.val_csv, splits_dir / "val.csv"))
    root_dir = resolve_project_path(args.root_dir)
    rgb_checkpoint = resolve_project_path(args.rgb_checkpoint) if args.rgb_checkpoint is not None else None
    fft_checkpoint = resolve_project_path(args.fft_checkpoint) if args.fft_checkpoint is not None else None
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    if rgb_checkpoint is None or fft_checkpoint is None:
        raise ValueError(
            "Fusion training requires pretrained RGB and FFT checkpoints. "
            "Provide both --rgb-checkpoint and --fft-checkpoint."
        )
    if not rgb_checkpoint.exists():
        raise FileNotFoundError(f"RGB checkpoint not found: {rgb_checkpoint}")
    if not fft_checkpoint.exists():
        raise FileNotFoundError(f"FFT checkpoint not found: {fft_checkpoint}")

    set_seed(seed)

    train_loader = build_dataloader(train_csv, root_dir, image_size, batch_size, num_workers, train=True)
    val_loader = build_dataloader(val_csv, root_dir, image_size, batch_size, num_workers, train=False)

    rgb_model = RGBStreamResNet(pretrained=False)
    fft_model = FFTStreamCNN()

    load_checkpoint(rgb_model, rgb_checkpoint, strict=args.strict_load)
    load_checkpoint(fft_model, fft_checkpoint, strict=args.strict_load)
    if args.fft_preprocessing == "fft2":
        fft_model.set_use_rfft(False)
    elif args.fft_preprocessing == "rfft2":
        fft_model.set_use_rfft(True)

    model = FusionModel(
        rgb_model=rgb_model,
        fft_model=fft_model,
        freeze_backbones=freeze_backbones,
    ).to(device)

    class_weights = compute_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    history: list[Dict[str, float]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Training on {device} with freeze_backbones={freeze_backbones}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")
    print(f"RGB checkpoint: {rgb_checkpoint}")
    print(f"FFT checkpoint: {fft_checkpoint}")
    print(f"FFT preprocessing: {'rfft2' if fft_model.uses_rfft else 'fft2'}")
    print(f"Early stopping patience: {early_stopping_patience}")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        save_checkpoint(output_dir / "fusion_last.pt", model, optimizer, scheduler, epoch, epoch_metrics)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "fusion_best.pt", model, optimizer, scheduler, epoch, epoch_metrics)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after epoch {epoch}. "
                f"Best validation loss: {best_val_loss:.4f}"
            )
            break

    history_path = output_dir / "fusion_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
