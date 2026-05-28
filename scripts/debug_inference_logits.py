import sys
import torch
from pathlib import Path
import cv2
import numpy as np

# ensure repo root on sys.path for `src` imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.api.utils.inference import load_fusion_model, _image_to_tensor
from src.models.fusion_model import FusionModel
from src.data.augmentation import get_eval_augmentation

ckpt = Path('results/checkpoints/fusion_best.pt')
print('ckpt', ckpt.exists())
model = load_fusion_model(ckpt, device='cpu', strict=True)
model.eval()

# sample image path - reuse one from tests/data
img_path = Path('data/processed/processed/FaceForensics++_C23/real/000/frame_000000_face_00.jpg')
if not img_path.exists():
    # fallback to any image in data/raw
    for p in Path('data').rglob('*.jpg'):
        img_path = p
        break
print('using', img_path)
img = cv2.imread(str(img_path))
if img is None:
    raise SystemExit('no image')
# naive full-frame resize to 224
aug = get_eval_augmentation(224)
res = aug(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img_hwc = res['image'] if isinstance(res, dict) else res
tensor = _image_to_tensor(img_hwc).unsqueeze(0)  # (1,3,H,W)
print('tensor shape', tensor.shape)

# compute rgb features
rgb_model = model.rgb_model
fft_model = model.fft_model
rgb_headless = model._rgb_headless_layers
fft_headless = model._fft_headless_layers

with torch.no_grad():
    rgb_feat = FusionModel._extract_features(rgb_model, tensor, model.rgb_feature_dim, rgb_headless, 'RGB')
    fft_feat = FusionModel._extract_features(fft_model, tensor.mean(1, keepdim=True), model.fft_feature_dim, fft_headless, 'FFT')
    print('rgb_feat shape', rgb_feat.shape, 'mean', float(rgb_feat.mean()), 'std', float(rgb_feat.std()), 'norm', float(rgb_feat.norm()))
    print('fft_feat shape', fft_feat.shape, 'mean', float(fft_feat.mean()), 'std', float(fft_feat.std()), 'norm', float(fft_feat.norm()))
    fused = torch.cat([rgb_feat, fft_feat], dim=1)
    print('fused mean/std/norm', float(fused.mean()), float(fused.std()), float(fused.norm()))
    logits = model.classifier(fused)
    print('logits', logits)
    probs = torch.softmax(logits, dim=1)
    print('probs', probs)
    # inspect classifier final layer
    for name, p in model.classifier.named_parameters():
        print('clf param', name, p.shape, float(p.mean()), float(p.std()), float(p.norm()))

print('done')
