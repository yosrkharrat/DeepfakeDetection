"""Claim verification endpoints."""

from __future__ import annotations

import time
from typing import Any

from flask import Blueprint, current_app, jsonify, request

claim_verify_bp = Blueprint("claim_verify", __name__)

MIN_CLAIM_LEN = 10
MAX_CLAIM_LEN = 5000
MAX_BATCH_SIZE = 20


@claim_verify_bp.route("/api/claim-verify", methods=["POST"])
def claim_verify():
    """Verify a claim (optionally with evidence) and return verdict + confidence."""
    data: dict[str, Any] = request.get_json(silent=True) or {}
    claim_value = data.get("claim")
    evidence_value = data.get("evidence")

    if not isinstance(claim_value, str):
        return (
            jsonify({"status": "error", "message": "Request body must contain a 'claim' field."}),
            400,
        )

    claim = claim_value.strip()
    if len(claim) < MIN_CLAIM_LEN:
        return (
            jsonify({"status": "error", "message": "Claim too short. Please provide more detail."}),
            400,
        )
    if len(claim) > MAX_CLAIM_LEN:
        return (
            jsonify({
                "status": "error",
                "message": f"Claim too long. Maximum {MAX_CLAIM_LEN} characters.",
            }),
            400,
        )

    evidence = None
    if evidence_value is not None:
        if not isinstance(evidence_value, str):
            return (
                jsonify({"status": "error", "message": "Evidence must be a string."}),
                400,
            )
        evidence = evidence_value.strip() or None

    model = current_app.config.get("CLAIM_VERIFY_MODEL")
    if model is None:
        return (
            jsonify({"status": "error", "message": "Claim verification model is not available."}),
            500,
        )

    start = time.perf_counter()
    try:
        result = model.predict(claim, evidence=evidence)
    except Exception as exc:
        return (
            jsonify({"status": "error", "message": f"Model inference failed: {exc}"}),
            500,
        )

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 1)
    response = {
        "status": "success",
        "media_type": "claim",
        "label": result["label"],
        "confidence": result["confidence"],
        "p_support": result["p_support"],
        "p_refute": result["p_refute"],
        "p_nei": result["p_nei"],
        "claim_length": len(claim),
        "elapsed_ms": elapsed_ms,
    }
    if evidence is not None:
        response["evidence_length"] = len(evidence)
    return jsonify(response)


@claim_verify_bp.route("/api/claim-verify/batch", methods=["POST"])
def claim_verify_batch():
    """Accept a list of claims and return per-claim verdicts."""
    data: dict[str, Any] = request.get_json(silent=True) or {}
    claims = data.get("claims")

    if not isinstance(claims, list):
        return (
            jsonify({"status": "error", "message": "Missing 'claims' array."}),
            400,
        )

    if len(claims) > MAX_BATCH_SIZE:
        return (
            jsonify({"status": "error", "message": f"Max {MAX_BATCH_SIZE} claims per batch."}),
            400,
        )

    model = current_app.config.get("CLAIM_VERIFY_MODEL")
    if model is None:
        return (
            jsonify({"status": "error", "message": "Claim verification model is not available."}),
            500,
        )

    results = []
    for i, item in enumerate(claims):
        if not isinstance(item, dict):
            results.append({"index": i, "status": "error", "message": "Each item must be an object with a 'claim' field."})
            continue

        claim_value = item.get("claim")
        if not isinstance(claim_value, str):
            results.append({"index": i, "status": "error", "message": "Missing 'claim' field."})
            continue

        claim = claim_value.strip()
        if len(claim) < MIN_CLAIM_LEN:
            results.append({"index": i, "status": "error", "message": "Claim too short."})
            continue
        if len(claim) > MAX_CLAIM_LEN:
            results.append({"index": i, "status": "error", "message": f"Claim too long (max {MAX_CLAIM_LEN})."})
            continue

        evidence_value = item.get("evidence")
        evidence = evidence_value.strip() if isinstance(evidence_value, str) and evidence_value.strip() else None

        try:
            result = model.predict(claim, evidence=evidence)
            results.append({"index": i, "status": "ok", **result})
        except Exception as exc:
            results.append({"index": i, "status": "error", "message": str(exc)})

    supported_count = sum(1 for r in results if r.get("label") == "SUPPORTED")
    refuted_count = sum(1 for r in results if r.get("label") == "REFUTED")
    return jsonify({
        "status": "success",
        "total": len(claims),
        "supported_count": supported_count,
        "refuted_count": refuted_count,
        "not_enough_info_count": len(claims) - supported_count - refuted_count,
        "results": results,
    })
