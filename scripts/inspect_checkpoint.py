import torch
from pathlib import Path
p = Path('rgb_finetune_epoch3.pth')
ckpt = torch.load(p, map_location='cpu')
if isinstance(ckpt, dict):
    for k in ('state_dict','model_state_dict','model','weights'):
        if k in ckpt and isinstance(ckpt[k], dict):
            ckpt = ckpt[k]
            break

if not isinstance(ckpt, dict):
    print('NOT_DICT', type(ckpt))
else:
    keys = list(ckpt.keys())
    print('NUM_KEYS', len(keys))
    for i,k in enumerate(keys[:120]):
        print(i, k)
    if len(keys)>120:
        print('...')
