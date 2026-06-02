"""Entry point — run from the project root: python run.py"""

import os
from src.api.app import create_app

device = os.environ.get("DEVICE", "cpu")
threshold = float(os.environ.get("THRESHOLD", "0.5"))

checkpoint = os.path.abspath("models/rgb_resnet18_best.pt")
mode = "rgb"

print(f"[run.py] MODE: {mode}")
print(f"[run.py] CHECKPOINT: {checkpoint}")

app = create_app(checkpoint_path=checkpoint, device=device, threshold=threshold, mode=mode)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)