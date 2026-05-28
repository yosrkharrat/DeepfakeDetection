from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import torch
from src.models.rgb_stream import RGBStreamResNet
from src.models.fft_stream import FFTStreamCNN
from src.models.fusion_model import FusionModel

out = Path("results/checkpoints")
out.mkdir(parents=True, exist_ok=True)

model = FusionModel(rgb_model=RGBStreamResNet(pretrained=False), fft_model=FFTStreamCNN(), freeze_backbones=True)
path = out / "fusion_best.pt"
torch.save(model.state_dict(), path)
print(f"Saved dummy fusion checkpoint to: {path}")
