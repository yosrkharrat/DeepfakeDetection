"""Evaluate a trained fusion checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import get_eval_augmentation
from src.data.dataset import DeepfakeDataset
from src.models.fft_stream import FFTStreamCNN
from src.models.fusion_model import FusionModel
from src.models.rgb_stream import RGBStreamResNet
from src.utils.metrics import binary_classification_metrics, positive_class_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RGB+FFT fusion model.")
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--root-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require checkpoint keys to match exactly when loading the fusion model.",
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


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not all(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint(model: nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}.")

    state_dict = strip_module_prefix(checkpoint)
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
) -> DataLoader:
    dataset = DeepfakeDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        transform=get_eval_augmentation(image_size),
        use_fft=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate_model(
    model: FusionModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, float | int | None]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels_all: list[int] = []
    scores_all: list[float] = []

    progress = tqdm(dataloader, desc="Test", leave=False)
    for batch in progress:
        if len(batch) == 3:
            rgb_batch, _, labels = batch
        else:
            rgb_batch, labels = batch
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(rgb_batch, rgb_batch)
        loss = criterion(logits, labels)
        fake_scores = positive_class_probabilities(logits)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        labels_all.extend(labels.detach().cpu().tolist())
        scores_all.extend(fake_scores.detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = binary_classification_metrics(labels_all, scores_all, threshold=threshold)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in the [0.0, 1.0] interval.")

    config = load_config(args.config)

    image_size = int(get_nested(config, "data", "image_size", default=224))
    splits_dir = resolve_project_path(get_nested(config, "data", "splits_dir", default="data/splits"))
    metrics_output = resolve_project_path(
        resolve_arg(args.output, get_nested(config, "paths", "metrics", default="results/metrics.json"))
    )
    batch_size = int(resolve_arg(args.batch_size, get_nested(config, "training", "batch_size", default=32)))
    num_workers = int(resolve_arg(args.num_workers, get_nested(config, "training", "num_workers", default=0)))
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

    test_csv = resolve_project_path(resolve_arg(args.test_csv, splits_dir / "test.csv"))
    root_dir = resolve_project_path(args.root_dir)
    fusion_checkpoint = resolve_project_path(args.fusion_checkpoint)
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    if not fusion_checkpoint.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {fusion_checkpoint}")

    dataloader = build_dataloader(test_csv, root_dir, image_size, batch_size, num_workers)

    rgb_model = RGBStreamResNet(pretrained=False)
    fft_model = FFTStreamCNN()
    model = FusionModel(rgb_model=rgb_model, fft_model=fft_model, freeze_backbones=True).to(device)
    load_checkpoint(model, fusion_checkpoint, strict=args.strict_load)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, dataloader, criterion, device, threshold=args.threshold)

    output = {
        "checkpoint": str(fusion_checkpoint),
        "test_csv": str(test_csv),
        "metrics": metrics,
    }

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved metrics to {metrics_output}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
