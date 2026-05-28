import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from src.api.utils.inference import FusionInferenceService

svc = FusionInferenceService.from_checkpoint(checkpoint_path='results/checkpoints/fusion_best.pt', device='cpu', strict=False)
root = Path('data/processed/processed/FaceForensics++_C23')
imgs = list(root.rglob('*.jpg'))[:12]
print('TEST_IMAGES', len(imgs))
for p in imgs:
    res = svc.predict_path(p)
    print(p.name, res['is_fake'], res['confidence'], res['num_faces'])
