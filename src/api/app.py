"""Flask application factory and entry point."""

import os

from flask import Flask, render_template

from src.api.routes.detect import detect_bp
from src.api.routes.health import health_bp
from src.api.utils.inference import FusionInferenceService


def create_app(
    checkpoint_path: str = "results/checkpoints/fusion_best.pt",
    device: str = "cpu",
    threshold: float = 0.5,
) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

    service = FusionInferenceService.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold,
    )
    app.config["INFERENCE_SERVICE"] = service

    app.register_blueprint(health_bp)
    app.register_blueprint(detect_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


