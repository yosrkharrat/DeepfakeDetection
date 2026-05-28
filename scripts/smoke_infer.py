import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from src.api.utils.inference import FusionInferenceService

ckpt = Path('results/checkpoints/fusion_best.pt')
img = Path('data/processed/processed/FaceForensics++_C23/real/000/frame_000000_face_00.jpg')
print('CKPT', ckpt.exists(), 'IMG', img.exists())
try:
    svc = FusionInferenceService.from_checkpoint(checkpoint_path=ckpt, device='cpu', strict=False)
    print('LOADED SERVICE')
    res = svc.predict_path(img)
    print('PREDICTION', res)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('ERROR', e)
