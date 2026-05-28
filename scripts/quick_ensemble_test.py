import os
import sys
import numpy as np
sys.path.insert(0, '.')
from src.api.utils.inference import FusionInferenceService

# Environment overrides
os.environ['FFT_CHECKPOINT'] = 'results/checkpoints/fft_extracted_from_fusion.pth'
rgb_ckpt = 'results/checkpoints/rgb_finetune_epoch3.pth'
if not os.path.exists(rgb_ckpt):
    rgb_ckpt = 'results/checkpoints/fusion_best.pt'
os.environ['RGB_CHECKPOINT'] = rgb_ckpt

service = FusionInferenceService.from_checkpoint('results/checkpoints/fusion_best.pt', device='cpu', mode='ensemble')

# Create 3 synthetic face crops (BGR uint8)
crops = [ (np.random.randint(0,256,(224,224,3),dtype=np.uint8)) for _ in range(3) ]
boxes = [(0,0,224,224)] * len(crops)

results = service._predict_crops(crops, boxes)
for i, r in enumerate(results):
    print(f'Face {i}: is_fake={r["is_fake"]}, confidence={r["confidence"]:.4f}, probs={r["probabilities"]}')
