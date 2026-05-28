import torch
from pathlib import Path
ckpt = Path('results/checkpoints/fusion_best.pt')
print('exists', ckpt.exists())
ck = torch.load(ckpt, map_location='cpu')
# unwrap
if isinstance(ck, dict):
    for k in ('state_dict','model_state_dict','model','weights'):
        if k in ck and isinstance(ck[k], dict):
            sd = ck[k]
            break
    else:
        sd = ck
else:
    sd = ck

keys = sorted(sd.keys())
print('num keys', len(keys))
for k in keys:
    if k.startswith('classifier') or k.endswith('.weight') and ('classifier' in k):
        print('key', k)

# interested keys
for name in ['classifier.0.weight','classifier.0.bias','classifier.3.weight','classifier.3.bias','classifier.5.weight','classifier.5.bias']:
    if name in sd:
        t = sd[name]
        print(name, 'shape', tuple(t.shape), 'mean', float(t.mean()), 'std', float(t.std()), 'norm', float(t.norm()))
    else:
        print(name, 'MISSING')

# Also print some other classifier-like keys
for k in keys:
    if 'classifier' in k and k not in ['classifier.0.weight','classifier.0.bias','classifier.3.weight','classifier.3.bias','classifier.5.weight','classifier.5.bias']:
        print('extra', k)

# Print some rgb backbone head if present
for k in keys:
    if k.startswith('rgb_model') and ('classifier' in k or 'head' in k or 'fc' in k or 'logits' in k):
        print('rgb key', k)

print('done')
