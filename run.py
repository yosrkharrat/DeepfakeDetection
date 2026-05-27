"""Entry point — run from the project root: python run.py"""

import os
from src.api.app import create_app

checkpoint = os.environ.get("CHECKPOINT", "results/checkpoints/fusion_best.pt")
device     = os.environ.get("DEVICE", "cpu")

app = create_app(checkpoint_path=checkpoint, device=device)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
