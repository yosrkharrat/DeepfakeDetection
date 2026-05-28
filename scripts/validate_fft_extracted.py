import sys
sys.path.insert(0, '.')
import torch
from src.models.fft_stream import FFTOnlyClassifier
sd = torch.load('results/checkpoints/fft_extracted_from_fusion.pth', map_location='cpu')
model = FFTOnlyClassifier()
# load_state_dict returns a NamedTuple (missing_keys, unexpected_keys) in newer PyTorch; use try/except
try:
    res = model.load_state_dict(sd, strict=False)
    print('Loaded into FFTOnlyClassifier (strict=False).')
    print(res)
except Exception as e:
    print('Load failed:', e)
