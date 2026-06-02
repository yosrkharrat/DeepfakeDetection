"""Flask application factory and entry point."""

import os
from pathlib import Path

from flask import Flask

from src.api.routes.claim_verify import claim_verify_bp
from src.api.routes.detect import detect_bp
from src.api.routes.fake_news import fake_news_bp
from src.api.routes.health import health_bp
from src.api.utils.inference import FusionInferenceService
from src.agents.claim_verifier import ClaimVerifier, is_ollama_available
from src.models.claim_verify_model import ClaimVerifyDetector
from src.models.fake_news_model import FakeNewsDetector


def create_app(
    checkpoint_path: str = "results/checkpoints/fusion_best.pt",
    device: str = "cpu",
    threshold: float = 0.5,
    mode: str = "fusion",
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run: python create_dummy_checkpoint.py  (random weights, for testing)\n"
            "Or set CHECKPOINT env var to the path of your trained .pt file."
        )

    service = FusionInferenceService.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold,
        mode=mode,
    )
    app.config["INFERENCE_SERVICE"] = service

    fake_news_checkpoint = os.environ.get("FAKE_NEWS_CHECKPOINT")
    if not fake_news_checkpoint:
        default_path = Path("results/checkpoints/fake_news_model")
        if default_path.exists():
            fake_news_checkpoint = str(default_path)
        else:
            # Public HF model — downloaded and cached on first run
            fake_news_checkpoint = "mrm8488/bert-tiny-finetuned-fake-news-detection"
    if fake_news_checkpoint:
        fake_news_device = os.environ.get("FAKE_NEWS_DEVICE", device)
        use_hf_inference = os.environ.get("FAKE_NEWS_USE_HF_INFERENCE", "0") in {"1", "true", "True"}
        hf_provider = os.environ.get("FAKE_NEWS_PROVIDER", "hf-inference")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        pos_label = os.environ.get("FAKE_NEWS_POS_LABEL")
        neg_label = os.environ.get("FAKE_NEWS_NEG_LABEL")
        try:
            app.config["FAKE_NEWS_MODEL"] = FakeNewsDetector(
                checkpoint_path=fake_news_checkpoint,
                device=fake_news_device,
                use_hf_inference=use_hf_inference,
                provider=hf_provider,
                api_key=hf_token,
                positive_label=pos_label,
                negative_label=neg_label,
            )
        except Exception as exc:
            app.logger.warning("Failed to load fake news model: %s", exc)
    else:
        app.logger.info("FAKE_NEWS_CHECKPOINT not set; fake news endpoint disabled.")

    claim_verify_checkpoint = os.environ.get("CLAIM_VERIFY_CHECKPOINT")
    if not claim_verify_checkpoint:
        default_path = Path("results/checkpoints/claim_verify_model")
        if default_path.exists():
            claim_verify_checkpoint = str(default_path)
        else:
            # Public NLI model — downloaded and cached on first run
            claim_verify_checkpoint = "cross-encoder/nli-deberta-v3-small"
    if claim_verify_checkpoint:
        claim_verify_device = os.environ.get("CLAIM_VERIFY_DEVICE", device)
        use_hf_inference = os.environ.get("CLAIM_VERIFY_USE_HF_INFERENCE", "0") in {"1", "true", "True"}
        hf_provider = os.environ.get("CLAIM_VERIFY_PROVIDER", "hf-inference")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        support_label = os.environ.get("CLAIM_VERIFY_SUPPORT_LABEL")
        refute_label = os.environ.get("CLAIM_VERIFY_REFUTE_LABEL")
        neutral_label = os.environ.get("CLAIM_VERIFY_NEUTRAL_LABEL")
        try:
            app.config["CLAIM_VERIFY_MODEL"] = ClaimVerifyDetector(
                checkpoint_path=claim_verify_checkpoint,
                device=claim_verify_device,
                use_hf_inference=use_hf_inference,
                provider=hf_provider,
                api_key=hf_token,
                support_label=support_label,
                refute_label=refute_label,
                neutral_label=neutral_label,
            )
        except Exception as exc:
            app.logger.warning("Failed to load claim verification model: %s", exc)
    else:
        app.logger.info("CLAIM_VERIFY_CHECKPOINT not set; claim verification endpoint disabled.")

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    if is_ollama_available(ollama_url):
        try:
            app.config["CLAIM_RESEARCH_AGENT"] = ClaimVerifier(
                ollama_base_url=ollama_url,
                max_iterations=int(os.environ.get("CLAIM_MAX_ITERATIONS", "1")),
            )
            app.logger.info("Multi-agent claim verifier loaded (Ollama at %s).", ollama_url)
        except Exception as exc:
            app.logger.warning("Failed to load claim research agent: %s", exc)
    else:
        app.logger.warning(
            "Ollama not reachable at %s — /api/research-claim will return 503. "
            "Start Ollama and restart the server to enable it.",
            ollama_url,
        )

    app.register_blueprint(health_bp)
    app.register_blueprint(detect_bp)
    app.register_blueprint(fake_news_bp)
    app.register_blueprint(claim_verify_bp)

    return app


