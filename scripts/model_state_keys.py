import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
import torch
from src.models.fusion_model import FusionModel
from src.models.rgb_stream import RGBStreamResNet
from src.models.fft_stream import FFTStreamCNN

model = FusionModel(rgb_model=RGBStreamResNet(pretrained=False), fft_model=FFTStreamCNN(), freeze_backbones=True)
keys = list(model.state_dict().keys())
print('NUM_MODEL_KEYS', len(keys))
for i,k in enumerate(keys[:200]):
    print(i, k)
print('...')
