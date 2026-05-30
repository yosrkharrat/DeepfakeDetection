"""POST /api/detect — image and video deepfake detection."""

import os
import tempfile
import time

try:
    import cv2
except Exception:  # pragma: no cover - runtime env dependent
    cv2 = None
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from src.api.utils.audio_inference import predict_audio_from_video
from src.api.utils.validators import validate_upload

detect_bp = Blueprint("detect", __name__)

MAX_VIDEO_FRAMES = 16
VIDEO_FRAME_STRIDE = 30


@detect_bp.post("/api/detect")
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    try:
        media_type, ext = validate_upload(file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    service = current_app.config["INFERENCE_SERVICE"]
    t0 = time.perf_counter()

    if media_type == "image":
        return _handle_image(file, service, t0)
    return _handle_video(file, ext, service, t0)


def _handle_image(file, service, t0):
    if cv2 is None:
        return jsonify({"error": "OpenCV is not installed on the server. Install opencv-python."}), 500
    data = np.frombuffer(file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Could not decode image."}), 400

    h, w = image_bgr.shape[:2]

    try:
        result = service.predict_image_array(image_bgr)
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    elapsed = time.perf_counter() - t0
    return jsonify({
        "media_type": "image",
        "original_width": w,
        "original_height": h,
        "elapsed_seconds": round(elapsed, 3),
        **result,
    })


def _handle_video(file, ext, service, t0):
    if cv2 is None:
        return jsonify({"error": "OpenCV is not installed on the server. Install opencv-python."}), 500
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    audio_result = None
    audio_error = None
    frame_results = []

    try:
        try:
            frame_results = _process_video(tmp_path, service)
        except Exception as exc:
            return jsonify({"error": f"Video processing failed: {exc}"}), 500
        if hasattr(service, "audio_model"):
            try:
                audio_result = predict_audio_from_video(service, tmp_path)
            except Exception as exc:
                audio_error = str(exc)
    finally:
        os.unlink(tmp_path)

    if not frame_results:
        return jsonify({"error": "No frames could be extracted from the video."}), 400

    fake_probs = [fr["best_fake_prob"] for fr in frame_results]
    avg_fake = float(np.mean(fake_probs))
    is_fake = avg_fake >= service.threshold
    confidence = avg_fake if is_fake else (1.0 - avg_fake)
    fake_frame_count = sum(1 for fr in frame_results if fr["is_fake"])

    combined = None
    if audio_result and "probabilities" in audio_result:
        weight_visual = float(os.environ.get("AV_WEIGHT_VISUAL", "0.6"))
        weight_audio = float(os.environ.get("AV_WEIGHT_AUDIO", "0.4"))
        total = weight_visual + weight_audio
        if total <= 0.0:
            total = 1.0
        p_fake_audio = float(audio_result["probabilities"]["fake"])
        p_combined = (weight_visual * avg_fake + weight_audio * p_fake_audio) / total
        combined_is_fake = p_combined >= service.threshold
        combined_confidence = p_combined if combined_is_fake else (1.0 - p_combined)
        combined = {
            "is_fake": bool(combined_is_fake),
            "confidence": round(float(combined_confidence), 4),
            "combined_fake_probability": round(float(p_combined), 4),
            "weights": {"visual": weight_visual, "audio": weight_audio},
        }

    elapsed = time.perf_counter() - t0
    response = {
        "media_type": "video",
        "is_fake": is_fake,
        "confidence": round(confidence, 4),
        "avg_fake_probability": round(avg_fake, 4),
        "num_frames_analyzed": len(frame_results),
        "fake_frame_count": fake_frame_count,
        "frame_results": frame_results,
        "elapsed_seconds": round(elapsed, 3),
    }
    if audio_result is not None:
        response["audio"] = audio_result
    if audio_error is not None:
        response["audio_error"] = audio_error
    if combined is not None:
        response["combined"] = combined
    return jsonify(response)


def _process_video(path: str, service) -> list[dict]:
    from flask import current_app
    cap = cv2.VideoCapture(path)
    results = []
    frame_idx = 0
    sampled = 0

    while sampled < MAX_VIDEO_FRAMES:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % VIDEO_FRAME_STRIDE == 0:
            try:
                result = service.predict_image_array(frame_bgr)
                best_fake = max(f["probabilities"]["fake"] for f in result["faces"])
                results.append({
                    "frame_index": frame_idx,
                    "is_fake": result["is_fake"],
                    "confidence": round(result["confidence"], 4),
                    "best_fake_prob": round(best_fake, 4),
                    "num_faces": result["num_faces"],
                })
                sampled += 1
            except Exception as exc:
                current_app.logger.warning("Frame %d skipped: %s", frame_idx, exc)
        frame_idx += 1

    cap.release()
    return results
