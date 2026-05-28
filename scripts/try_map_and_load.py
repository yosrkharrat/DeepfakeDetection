import torch
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from src.models.fusion_model import FusionModel
from src.models.rgb_stream import RGBStreamResNet
from src.models.fft_stream import FFTStreamCNN

ckpt_path = Path('rgb_finetune_epoch3.pth')
ckpt = torch.load(ckpt_path, map_location='cpu')
if isinstance(ckpt, dict):
    for k in ('state_dict','model_state_dict','model','weights'):
        if k in ckpt and isinstance(ckpt[k], dict):
            ckpt = ckpt[k]
            break
if not isinstance(ckpt, dict):
    print('CKPT NOT DICT', type(ckpt)); raise SystemExit

ckpt_keys = list(ckpt.keys())
print('CKPT_KEYS', len(ckpt_keys))

model = FusionModel(rgb_model=RGBStreamResNet(pretrained=False), fft_model=FFTStreamCNN(), freeze_backbones=True)
model_keys = list(model.state_dict().keys())
print('MODEL_KEYS', len(model_keys))

# Attempt mapping: for each ck key, try to find model key that endswith ck_key after removing leading numeric group
mapped = {}
unmatched = []
for ck in ckpt_keys:
    parts = ck.split('.', 1)
    if len(parts)==1:
        suffix = parts[0]
    else:
        suffix = parts[1]
    matches = [mk for mk in model_keys if mk.endswith(suffix)]
    if len(matches)==1:
        mapped[matches[0]] = ckpt[ck]
    else:
        unmatched.append((ck, len(matches)))

print('MAPPED_KEYS', len(mapped))
print('UNMATCHED_KEYS', len(unmatched))
for i,u in enumerate(unmatched[:30]):
    print(i, u)

# Try loading mapped state dict
new_state = {k:v for k,v in mapped.items()}
missing_keys, unexpected_keys = model.load_state_dict(new_state, strict=False)
print('AFTER_LOAD missing', len(missing_keys), 'unexpected', len(unexpected_keys))
print('SAMPLE_MISSING', missing_keys[:20])
