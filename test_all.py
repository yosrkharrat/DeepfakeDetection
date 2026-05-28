"""
DeepGuard - full test suite
Usage:
    python test_all.py
    python test_all.py --checkpoint path/to/fusion_best.pt
"""
import sys, os, time, json, base64, subprocess, threading, argparse
import numpy as np

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"; BOLD = "\033[1m"
ok   = lambda msg: (print(f"  [PASS] {msg}"), RESULTS.append(True))
fail = lambda msg: (print(f"  [FAIL] {msg}"), RESULTS.append(False))
hdr  = lambda msg: print(f"\n-- {msg} " + "-"*(50-len(msg)))
RESULTS = []

sys.path.insert(0, os.path.dirname(__file__))

# ── parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="results/checkpoints/fusion_best.pt")
args = parser.parse_args()
CHECKPOINT = args.checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dependencies
# ─────────────────────────────────────────────────────────────────────────────
hdr("1. Python dependencies")
try:
    import torch, torchvision, cv2, flask, albumentations, facenet_pytorch
    ok("core packages (torch, torchvision, cv2, flask, albumentations, facenet_pytorch)")
except ImportError as e:
    fail(f"core package missing: {e}")

try:
    from pytorch_grad_cam import EigenCAM
    ok("grad-cam / pytorch_grad_cam")
except ImportError:
    fail("grad-cam not installed - run: pip install grad-cam")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model imports & feature dims
# ─────────────────────────────────────────────────────────────────────────────
hdr("2. Model imports & feature dims")
try:
    from src.models.fft_stream   import FFTStreamCNN, compute_fft_magnitude
    from src.models.rgb_stream   import RGBStreamResNet
    from src.models.fusion_model import FusionModel
    ok("all model modules importable")
except Exception as e:
    fail(f"model import error: {e}"); sys.exit(1)

try:
    fft_m = FFTStreamCNN()
    rgb_m = RGBStreamResNet(pretrained=False)
    assert fft_m.feature_dim == 256,  f"got {fft_m.feature_dim}"
    assert rgb_m.feature_dim == 1792, f"got {rgb_m.feature_dim}"
    ok("FFTStreamCNN.feature_dim=256, RGBStreamResNet.feature_dim=1792")
except Exception as e:
    fail(f"feature_dim: {e}")

try:
    fusion_m = FusionModel(rgb_model=RGBStreamResNet(pretrained=False), fft_model=FFTStreamCNN())
    assert fusion_m.fused_feature_dim == 2048, f"got {fusion_m.fused_feature_dim}"
    ok(f"FusionModel.fused_feature_dim=2048")
except Exception as e:
    fail(f"fused_feature_dim: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Forward pass
# ─────────────────────────────────────────────────────────────────────────────
hdr("3. Model forward pass")
try:
    model = FusionModel(RGBStreamResNet(pretrained=False), FFTStreamCNN())
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 2), f"wrong shape: {out.shape}"
    ok(f"output shape (2, 2) logits OK")
except Exception as e:
    fail(f"forward pass: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FFT preprocessing
# ─────────────────────────────────────────────────────────────────────────────
hdr("4. FFT compute_fft_magnitude")
try:
    x = torch.rand(2, 3, 224, 224)
    s = compute_fft_magnitude(x)
    assert tuple(s.shape) == (2, 1, 224, 224), f"bad shape: {s.shape}"
    assert float(s.min()) >= 0 and float(s.max()) <= 1.0, f"bad range"
    ok(f"spectrum shape (2,1,224,224), values in [0,1]")
except Exception as e:
    fail(f"FFT preprocessing: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
hdr("5. Checkpoint")
if not os.path.exists(CHECKPOINT):
    print(f"  No checkpoint at {CHECKPOINT} - creating dummy...")
    result = subprocess.run([sys.executable, "create_dummy_checkpoint.py"], capture_output=True, text=True)
    print(f"  {result.stdout.strip()}")

if os.path.exists(CHECKPOINT):
    size_mb = os.path.getsize(CHECKPOINT) / 1024 / 1024
    ok(f"checkpoint exists ({size_mb:.1f} MB): {CHECKPOINT}")
else:
    fail(f"checkpoint missing: {CHECKPOINT}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Inference service
# ─────────────────────────────────────────────────────────────────────────────
hdr("6. Inference service")
try:
    from src.api.utils.inference import FusionInferenceService
    svc = FusionInferenceService.from_checkpoint(CHECKPOINT, threshold=0.5)
    np.random.seed(0)
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    r = svc.predict_image_array(img)
    assert "is_fake" in r and "faces" in r and "confidence" in r
    assert "gradcam_heatmap" in r["faces"][0], "gradcam_heatmap key missing"
    ok(f"predict_image_array: is_fake={r['is_fake']}, confidence={r['confidence']:.4f}, gradcam_heatmap key present")
except Exception as e:
    fail(f"inference service: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EigenCAM heatmap (force fake path)
# ─────────────────────────────────────────────────────────────────────────────
hdr("7. EigenCAM heatmap (fake=True path)")
try:
    svc2 = FusionInferenceService.from_checkpoint(CHECKPOINT, threshold=0.45)
    img2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    r2 = svc2.predict_image_array(img2)
    face2 = r2["faces"][0]
    assert face2["is_fake"], "expected fake at threshold=0.45 with dummy weights"
    hm = face2["gradcam_heatmap"]
    assert hm and len(hm) > 1000, f"heatmap empty: {hm!r}"
    img_bytes = base64.b64decode(hm)
    assert img_bytes[:2] == b'\xff\xd8', "not a valid JPEG"
    ok(f"heatmap generated: {len(hm)} base64 chars, {len(img_bytes)} bytes valid JPEG")
except Exception as e:
    fail(f"EigenCAM: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Flask endpoints
# ─────────────────────────────────────────────────────────────────────────────
hdr("8. Flask endpoints (starting server...)")
env = os.environ.copy()
env["CHECKPOINT"] = CHECKPOINT

server = subprocess.Popen(
    [sys.executable, "run.py"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    env=env,
)

# Wait up to 15 s for server to accept connections
import urllib.request, urllib.error
for _ in range(15):
    time.sleep(1)
    try:
        urllib.request.urlopen("http://localhost:5000/api/health")
        break
    except Exception:
        pass

try:
    import requests

    # GET /api/health
    r = requests.get("http://localhost:5000/api/health", timeout=5)
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    ok('GET /api/health -> {"status":"ok"}')

    # GET /api/info
    r = requests.get("http://localhost:5000/api/info", timeout=5)
    assert r.status_code == 200
    info = r.json()
    assert "EfficientNet-B4" in info.get("architecture", ""), f"wrong arch string"
    ok("GET /api/info: architecture contains EfficientNet-B4")

    # GET /
    r = requests.get("http://localhost:5000/", timeout=5)
    assert r.status_code == 200
    ok("GET / -> 200 (index.html)")

    # POST /api/detect
    np.random.seed(42)
    img3 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img3)
    r = requests.post(
        "http://localhost:5000/api/detect",
        files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")},
        timeout=30,
    )
    assert r.status_code == 200
    d = r.json()
    assert "is_fake" in d and "faces" in d and "gradcam_heatmap" in d["faces"][0]
    ok(f'POST /api/detect -> HTTP 200, is_fake={d["is_fake"]}, gradcam_heatmap key present')

except Exception as e:
    fail(f"Flask endpoint: {e}")
finally:
    server.terminate()
    server.wait()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(RESULTS)
failed = len(RESULTS) - passed
print(f"\n{'='*47}")
print(f"  {GREEN}PASSED: {passed}{RESET}   {RED if failed else ''}FAILED: {failed}{RESET if failed else ''}")
print(f"{'='*47}")
if failed == 0:
    print(f"  {GREEN}All tests passed.{RESET}")
else:
    print(f"  {RED}{failed} test(s) failed - see [FAIL] lines above.{RESET}")
sys.exit(0 if failed == 0 else 1)
