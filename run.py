"""Entry point — run from the project root: python run.py"""

import os
from src.api.app import create_app

checkpoint = os.environ.get("CHECKPOINT", "results/checkpoints/fusion_best.pt")
device     = os.environ.get("DEVICE", "cpu")
threshold  = float(os.environ.get("THRESHOLD", "0.5"))
mode       = os.environ.get("MODE", "fusion")

app = create_app(checkpoint_path=checkpoint, device=device, threshold=threshold, mode=mode)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
