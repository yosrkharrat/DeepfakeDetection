"""Health and info endpoints."""

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@health_bp.get("/api/info")
def info():
    return jsonify({
        "model": "Dual-Stream Fusion (RGB + FFT)",
        "architecture": "ResNet-18 + FFTStreamCNN → MLP classifier",
        "dataset": "FaceForensics++ C23",
        "test_accuracy": 0.9587,
        "test_auc_roc": 0.9951,
        "test_f1": 0.9690,
        "supported_inputs": ["image/jpeg", "image/png", "video/mp4"],
    })
