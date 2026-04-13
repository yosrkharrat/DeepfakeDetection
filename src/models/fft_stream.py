"""
FFT Stream — Deepfake Detection System
Frequency-domain CNN that detects GAN upsampling artifacts.

How it works:
  1. Apply 2D FFT to the input face crop (grayscale)
  2. Compute the log-magnitude spectrum (shifts DC component to center)
  3. Feed the spectrum into a lightweight CNN
  4. Output a 256-dim feature vector for the fusion classifier

Why this works:
  GANs generate images by repeatedly upsampling a small latent map.
  Each upsampling step leaves periodic grid-aligned spikes in the
  frequency domain — invisible to the eye, but mathematically distinct
  from real camera images.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FFT preprocessing (standalone utility — used in dataset.py too)
# ---------------------------------------------------------------------------

def compute_fft_magnitude(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the log-magnitude FFT spectrum of a face crop.

    Args:
        image_tensor: Float tensor of shape (B, C, H, W) in [0, 1].
                      If RGB (C=3), converts to grayscale before FFT.

    Returns:
        Spectrum tensor of shape (B, 1, H, W), values in [0, 1].
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Convert to grayscale if needed (GAN artifacts are channel-independent)
    if image_tensor.shape[1] == 3:
        # Standard luminance weights
        gray = (
            0.2989 * image_tensor[:, 0]
            + 0.5870 * image_tensor[:, 1]
            + 0.1140 * image_tensor[:, 2]
        ).unsqueeze(1)
    else:
        gray = image_tensor

    # 2D FFT via PyTorch (runs on GPU if tensor is on GPU)
    fft = torch.fft.fft2(gray)

    # Shift DC component to center of spectrum
    fft_shifted = torch.fft.fftshift(fft)

    # Log-magnitude (log1p avoids log(0), compresses dynamic range)
    magnitude = torch.log1p(torch.abs(fft_shifted))

    # Normalize to [0, 1] per image
    b = magnitude.shape[0]
    mag_flat = magnitude.view(b, -1)
    mag_min = mag_flat.min(dim=1).values.view(b, 1, 1, 1)
    mag_max = mag_flat.max(dim=1).values.view(b, 1, 1, 1)
    magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)

    return magnitude  # shape: (B, 1, H, W)


# ---------------------------------------------------------------------------
# FFT CNN architecture
# ---------------------------------------------------------------------------

class FFTBlock(nn.Module):
    """
    Basic conv block: Conv → BN → ReLU → Conv → BN → ReLU
    with an optional residual shortcut.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut projection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class FFTStreamCNN(nn.Module):
    """
    Lightweight CNN that processes the log-magnitude FFT spectrum
    of a 224×224 face crop and outputs a 256-dim feature vector.

    Architecture:
        Input : (B, 1, 224, 224) — single-channel spectrum
        Stem  : Conv 7×7 → BN → ReLU → MaxPool  → (B, 32, 56, 56)
        Block1: FFTBlock(32→64,  stride=2)        → (B, 64, 28, 28)
        Block2: FFTBlock(64→128, stride=2)        → (B, 128, 14, 14)
        Block3: FFTBlock(128→256,stride=2)        → (B, 256, 7, 7)
        Block4: FFTBlock(256→256,stride=2)        → (B, 256, 4, 4)
        Pool  : AdaptiveAvgPool → (B, 256, 1, 1)
        Head  : Flatten → Dropout(0.3) → Linear(256, 256)
        Output: (B, 256)

    The output feeds into the fusion FC classifier alongside the RGB stream.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        # Stem: large receptive field to capture frequency patterns early
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # → 56×56
        )

        # Residual blocks — progressively downsample
        self.block1 = FFTBlock(32,  64,  stride=2)   # → 28×28
        self.block2 = FFTBlock(64,  128, stride=2)   # → 14×14
        self.block3 = FFTBlock(128, 256, stride=2)   # →  7×7
        self.block4 = FFTBlock(256, 256, stride=2)   # →  4×4

        # Global average pooling + projection head
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.head    = nn.Linear(256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw RGB face crop (B, 3, 224, 224) OR
               pre-computed spectrum (B, 1, 224, 224).
               If RGB is passed, FFT is computed internally.

        Returns:
            Feature vector of shape (B, 256).
        """
        # Accept raw RGB input for convenience during inference
        if x.shape[1] == 3:
            x = compute_fft_magnitude(x)

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.head(x)
        return x  # (B, 256)


# ---------------------------------------------------------------------------
# Standalone deepfake classifier (FFT stream only, for ablation study)
# ---------------------------------------------------------------------------

class FFTOnlyClassifier(nn.Module):
    """
    Wraps FFTStreamCNN with a simple 2-class head.
    Use this for ablation experiments to measure FFT stream in isolation
    before integrating into the dual-stream fusion model.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone = FFTStreamCNN(dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)  # (B, 2) logits


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== FFT Stream sanity check ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Test FFT preprocessing ---
    dummy_rgb = torch.rand(4, 3, 224, 224).to(device)
    spectrum  = compute_fft_magnitude(dummy_rgb)
    print(f"FFT spectrum shape : {spectrum.shape}")     # (4, 1, 224, 224)
    print(f"Spectrum range     : [{spectrum.min():.3f}, {spectrum.max():.3f}]")

    # --- Test FFT CNN (spectrum input) ---
    model = FFTStreamCNN().to(device)
    features = model(spectrum)
    print(f"Feature vector shape: {features.shape}")   # (4, 256)

    # --- Test FFT CNN (raw RGB input — FFT computed internally) ---
    features2 = model(dummy_rgb)
    print(f"Feature vector (RGB in) shape: {features2.shape}")

    # --- Test standalone classifier ---
    clf = FFTOnlyClassifier().to(device)
    logits = clf(dummy_rgb)
    print(f"Classifier logits shape: {logits.shape}")  # (4, 2)
    probs = torch.softmax(logits, dim=1)
    print(f"Sample probs (real, fake): {probs[0].detach().cpu().numpy().round(3)}")

    # --- Parameter count ---
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFFTStreamCNN trainable params: {n_params:,}")

    print("\nAll checks passed.")
