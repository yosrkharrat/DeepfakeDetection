"""Text fake news detection endpoints."""

from __future__ import annotations

import time
from typing import Any

from flask import Blueprint, current_app, jsonify, request

fake_news_bp = Blueprint("fake_news", __name__)

MIN_TEXT_LEN = 20
MAX_TEXT_LEN = 10_000
MAX_BATCH_SIZE = 20


@fake_news_bp.route("/api/detect-text", methods=["POST"])
def detect_fake_news():
    """Accept JSON body with a 'text' field and return verdict + confidence."""
    data: dict[str, Any] = request.get_json(silent=True) or {}
    text_value = data.get("text")

    if not isinstance(text_value, str):
        return (
            jsonify({"status": "error", "message": "Request body must contain a 'text' field."}),
            400,
        )

    text = text_value.strip()
    if len(text) < MIN_TEXT_LEN:
        return (
            jsonify({
                "status": "error",
                "message": "Text too short. Please provide at least a full sentence.",
            }),
            400,
        )

    if len(text) > MAX_TEXT_LEN:
        return (
            jsonify({
                "status": "error",
                "message": f"Text too long. Maximum {MAX_TEXT_LEN} characters.",
            }),
            400,
        )

    model = current_app.config.get("FAKE_NEWS_MODEL")
    if model is None:
        return (
            jsonify({"status": "error", "message": "Fake news model is not available."}),
            500,
        )

    start = time.perf_counter()
    try:
        result = model.predict(text)
    except Exception as exc:
        return (
            jsonify({"status": "error", "message": f"Model inference failed: {exc}"}),
            500,
        )

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 1)
    return jsonify({
        "status": "success",
        "media_type": "text",
        "is_fake": result["is_fake"],
        "label": result["label"],
        "confidence": result["confidence"],
        "p_fake": result["p_fake"],
        "p_real": result["p_real"],
        "text_length": len(text),
        "elapsed_ms": elapsed_ms,
    })


@fake_news_bp.route("/api/detect-text/batch", methods=["POST"])
def detect_fake_news_batch():
    """Accept a list of texts and return per-text verdicts."""
    data: dict[str, Any] = request.get_json(silent=True) or {}
    texts = data.get("texts")

    if not isinstance(texts, list):
        return (
            jsonify({"status": "error", "message": "Missing 'texts' array."}),
            400,
        )

    if len(texts) > MAX_BATCH_SIZE:
        return (
            jsonify({"status": "error", "message": f"Max {MAX_BATCH_SIZE} texts per batch."}),
            400,
        )

    model = current_app.config.get("FAKE_NEWS_MODEL")
    if model is None:
        return (
            jsonify({"status": "error", "message": "Fake news model is not available."}),
            500,
        )

    results = []
    for i, text_value in enumerate(texts):
        if not isinstance(text_value, str):
            results.append({"index": i, "status": "error", "message": "Text must be a string."})
            continue
        text = text_value.strip()
        if len(text) < MIN_TEXT_LEN:
            results.append({"index": i, "status": "error", "message": "Text too short."})
            continue
        if len(text) > MAX_TEXT_LEN:
            results.append({
                "index": i,
                "status": "error",
                "message": f"Text too long (max {MAX_TEXT_LEN}).",
            })
            continue

        try:
            result = model.predict(text)
            results.append({"index": i, "status": "ok", **result})
        except Exception as exc:
            results.append({"index": i, "status": "error", "message": str(exc)})

    fake_count = sum(1 for r in results if r.get("is_fake"))
    return jsonify({
        "status": "success",
        "total": len(texts),
        "fake_count": fake_count,
        "real_count": len(texts) - fake_count,
        "results": results,
    })
