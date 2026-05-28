import torch
from pathlib import Path

CKPT_IN = Path('results/checkpoints/fusion_best.pt')
CKPT_OUT = Path('results/checkpoints/fft_extracted_from_fusion.pth')

ck = torch.load(CKPT_IN, map_location='cpu')
if isinstance(ck, dict) and 'state_dict' in ck:
    sd = ck['state_dict']
else:
    sd = ck

new_sd = {}
for k, v in sd.items():
    if k.startswith('fft_model.'):
        sub = k[len('fft_model.'):]
        if sub.startswith('backbone.'):
            target = sub
        else:
            target = 'backbone.' + sub
        new_sd[target] = v
    # also handle keys that are already like 'backbone.'
    elif k.startswith('backbone.') and 'fft' in k.lower():
        new_sd[k] = v

print(f'Found {len(new_sd)} FFT-related keys to save.')

# Save extracted state dict
CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(new_sd, CKPT_OUT)
print(f'Saved extracted FFT checkpoint to {CKPT_OUT}')

# Quick validation: try to load into FFTOnlyClassifier
try:
    from src.models.fft_stream import FFTOnlyClassifier
    model = FFTOnlyClassifier()
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print('Loaded into FFTOnlyClassifier with strict=False')
    print('Missing keys:', len([k for k in missing]))
    print('Unexpected keys:', len([k for k in unexpected]))
except Exception as e:
    print('Validation load failed:', e)
