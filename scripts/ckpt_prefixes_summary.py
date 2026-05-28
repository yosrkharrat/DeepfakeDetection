import torch
from pathlib import Path
p=Path('rgb_finetune_epoch3.pth')
ckpt=torch.load(p, map_location='cpu')
if isinstance(ckpt, dict):
    for k in ('state_dict','model_state_dict','model','weights'):
        if k in ckpt and isinstance(ckpt[k], dict):
            ckpt = ckpt[k]
            break
if not isinstance(ckpt, dict):
    print('NOT_DICT', type(ckpt))
    raise SystemExit
keys = list(ckpt.keys())
from collections import Counter
prefixes = [k.split('.')[0] for k in keys]
ctr = Counter(prefixes)
items = list(ctr.items())
items.sort(key=lambda x: (-x[1], x[0]))
for pref, cnt in items[:50]:
    print(pref, cnt)
print('TOTAL_KEYS', len(keys))
