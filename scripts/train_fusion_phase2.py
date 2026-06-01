"""
Phase 2 end-to-end fine-tuning for the dual-stream fusion model.

Loads the best Phase 1 checkpoint, unfreezes all backbones, and fine-tunes
with per-layer learning rates (backbone: 1e-5, classifier: 1e-4).

Usage:
    python scripts/train_fusion_phase2.py \
        --phase1_ckpt results/checkpoints/fusion_best.pt \
        --data_celeb  data/celeb-df \
        --epochs 10
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.metrics import roc_auc_score

from src.models.fusion_model import FusionModel
from src.models.rgb_stream import RGBStreamResNet
from src.models.fft_stream import FFTStreamCNN
from src.data.celebdf_dataset import CelebDFDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results/checkpoints", exist_ok=True)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for rgb, labels in loader:
            rgb = rgb.to(device)
            logits = model(rgb, rgb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    if len(set(all_labels)) < 2:
        return 0.5
    return roc_auc_score(all_labels, all_probs)


def train_phase2(model, train_loader, val_loader, epochs=10):
    # Unfreeze everything
    model.set_backbone_trainable(trainable=True)

    optimizer = torch.optim.Adam([
        {"params": model.rgb_model.parameters(), "lr": 1e-5},
        {"params": model.fft_model.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))

    best_auc = 0.0
    patience = 0

    for epoch in range(epochs):
        model.train()
        for rgb, labels in train_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(rgb, rgb), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        auc = evaluate(model, val_loader)
        print(f"[Phase 2] Epoch {epoch+1}/{epochs}  AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "results/checkpoints/fusion_best.pt")
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Phase 2 done. Best AUC: {best_auc:.4f}")
    print("Saved: results/checkpoints/fusion_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_ckpt", default="results/checkpoints/fusion_best.pt")
    parser.add_argument("--data_celeb",  default="data/celeb-df")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--epochs",      type=int, default=10)
    args = parser.parse_args()

    dataset = CelebDFDataset(args.data_celeb)
    if len(dataset) == 0:
        print(f"No images found in {args.data_celeb}. Add real/ and fake/ subfolders.")
        return

    val_size   = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    rgb_model = RGBStreamResNet(pretrained=True)
    fft_model = FFTStreamCNN()
    model = FusionModel(rgb_model, fft_model, freeze_backbones=True).to(device)

    if Path(args.phase1_ckpt).exists():
        model.load_state_dict(torch.load(args.phase1_ckpt, map_location=device))
        print(f"Loaded Phase 1 checkpoint: {args.phase1_ckpt}")
    else:
        print(f"No checkpoint found at {args.phase1_ckpt} — starting from scratch.")

    train_phase2(model, train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
