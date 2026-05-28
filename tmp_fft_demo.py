from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# Sample image path found earlier
img_path = Path('data/processed/processed/FaceForensics++_C23/real/000/frame_000000_face_00.jpg')
if not img_path.exists():
    raise SystemExit(f"Sample image not found: {img_path}")

img = Image.open(img_path).convert('RGB')
img_np = np.array(img)

# compute FFT magnitude
x = torch.from_numpy(img_np).permute(2,0,1).float()/255.0
x = x.unsqueeze(0)  # (1,C,H,W)
fft = torch.fft.fft2(x, dim=(-2,-1))
fft_shift = torch.roll(fft, shifts=(x.shape[-2]//2, x.shape[-1]//2), dims=(-2,-1))
mag = torch.log1p(torch.abs(fft_shift)).squeeze(0)

# visualize first channel or rgb channels
fft_vis = mag[:3].permute(1,2,0).cpu().numpy()
fft_vis = fft_vis - fft_vis.min()
fft_vis = fft_vis / (fft_vis.max()+1e-8)

out_dir = Path('results/plots')
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].imshow(img_np)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(fft_vis, cmap='magma')
axes[1].set_title('FFT magnitude')
axes[1].axis('off')

plt.tight_layout()
out_path = out_dir / 'fft_demo.png'
plt.savefig(out_path, dpi=150)
print('Saved FFT demo to', out_path)
