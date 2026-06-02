"""Entry point — run from the project root: python run.py"""

import os
from src.api.app import create_app

device    = os.environ.get("DEVICE", "cpu")
threshold = float(os.environ.get("THRESHOLD", "0.5"))

_rgb_ckpt = os.path.abspath("models/rgb_resnet18_best.pt")
_fft_ckpt = os.path.abspath("models/fft_model_v2.pth")

if os.path.exists(_rgb_ckpt) and os.path.exists(_fft_ckpt) and "MODE" not in os.environ:
    os.environ.setdefault("RGB_CHECKPOINT", _rgb_ckpt)
    os.environ.setdefault("FFT_CHECKPOINT", _fft_ckpt)
    checkpoint = os.environ.get("CHECKPOINT", _rgb_ckpt)
    mode = "ensemble"
else:
    checkpoint = os.environ.get("CHECKPOINT", "results/checkpoints/fusion_best.pt")
    mode = os.environ.get("MODE", "fusion")

app = create_app(checkpoint_path=checkpoint, device=device, threshold=threshold, mode=mode)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
