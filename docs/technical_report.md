# Reality Check / DeepGuard Technical Report

## 1. Executive Summary

This workspace contains a multi-part media credibility platform centered on a deepfake detection system, with additional text-based misinformation and claim-verification features. The main production-facing application is a Flask service branded in the UI as **DeepGuard**, while the core model stack is a dual-stream deepfake classifier that combines **RGB spatial features** and **FFT frequency features**. The platform also includes two broader intelligence integrations: a fake-news detection path inspired by the separate `fake_content_detection` repository, and a claim-research workflow backed by the `multi-agent-research-system` repository.

At a high level, the project is organized around three distinct analysis modes:

1. Visual deepfake detection for images and videos.
2. Text fake-news detection for pasted articles or headlines.
3. Claim verification and longer-form research synthesis through a multi-agent system.

The implementation emphasizes practical deployment: upload validation, face detection fallback paths, checkpoint compatibility handling, calibrated thresholds, optional audio analysis, a browser-based UI, and a small API surface that can be exercised from both the frontend and external tools.

---

## 2. Project Scope

The repository is not just a model script. It is a full application stack with the following layers:

- Data preparation and preprocessing for face crops.
- RGB and FFT model backbones.
- A fusion classifier for binary fake/real prediction.
- Inference services for images, videos, text, and claims.
- A Flask web UI and JSON API.
- Training scripts, evaluation utilities, and checkpoint loaders.
- External repo integrations for fake-news analysis and multi-agent claim research.

The repo also contains documentation, notebooks, scripts, and results artifacts that show the project evolved from model experimentation into a multi-modal application.

---

## 3. Repository Structure

The workspace is organized around a conventional Python application layout:

- `run.py` starts the Flask application.
- `src/api/` contains the application factory, routes, templates, static assets, and utility functions.
- `src/models/` contains the model backbones and wrappers.
- `src/data/` contains dataset loading, augmentation, and face detection.
- `src/training/` contains training logic and loss functions.
- `src/utils/` contains metrics, FFT helpers, visualization, Grad-CAM, and configuration helpers.
- `scripts/` contains offline utilities such as training, evaluation, export, inspection, and dataset preparation.
- `configs/` contains YAML configuration files.
- `results/` stores checkpoints, plots, and metrics.
- `docs/` contains project documentation.
- `notebooks/` contains EDA and training notebooks.

There is also a `frontend/` folder with placeholder files, but the active web UI is served from `src/api/templates/index.html` and `src/api/static/`.

---

## 4. Tech Stack

### Core runtime

- Python 3.x.
- Flask for the application server.
- PyTorch for model definition, loading, and inference.
- Torchvision for pretrained vision backbones.
- OpenCV for media decoding, video sampling, and image handling.
- Albumentations for image augmentation.
- Pillow for image IO.
- NumPy and pandas for array/data manipulation.
- PyYAML for configuration parsing.
- tqdm for training progress reporting.

### Model and ML tooling

- facenet-pytorch for MTCNN face detection when available.
- Torchvision EfficientNet-B4 as the primary RGB backbone.
- A custom FFT CNN for frequency-domain feature extraction.
- Transformers and huggingface_hub for text classification and remote inference paths.
- grad-cam for visual explanations in the UI.

### Optional audio and multimodal support

- MFCC-based CNN audio classifier.
- Wav2Vec2-based audio classifier wrapper.
- Audio extraction utilities for video-to-audio analysis.

### External orchestration stack

- LangGraph for multi-agent control flow.
- LangChain and ChatOllama for local LLM interaction.
- Ollama as the local model server required by the research agent system.

### Presentation layer

- HTML, CSS, and vanilla JavaScript.
- A compact tabbed UI for visual detection, fake-news detection, and claim verification.

### Storage and artifacts

- Local checkpoint files under `results/checkpoints/`.
- Metrics and calibration outputs under `results/`.
- Plots under `results/plots/`.

---

## 5. Application Overview

The main application entry point is `run.py`, which reads environment variables and starts the Flask factory in `src/api/app.py`.

The runtime supports several operational modes:

- `fusion` for the RGB + FFT deepfake model.
- `rgb` for RGB-only classification.
- `fft` for FFT-only classification.
- `audio` for audio-only inference.
- `ensemble` and `av_ensemble` for broader experimental combinations.

The UI exposes three tabs:

- Visual deepfake detection for images and videos.
- Fake news detection for pasted text.
- Claim verification for factual claims.

The project is therefore best understood as a credibility platform rather than a single deepfake classifier.

---

## 6. Deepfake Detection Architecture

### 6.1 High-level design

The core deepfake detector uses a **dual-stream fusion architecture**:

- The RGB branch captures spatial texture, shape, and appearance anomalies.
- The FFT branch captures frequency artifacts that often emerge from generation, interpolation, or upsampling.
- The fusion head combines both embeddings and produces a binary real/fake decision.

This design is intended to reduce the blind spots of a single-stream detector. RGB features are strong on local appearance cues, while FFT features are useful for periodic artifacts and latent structural signals.

### 6.2 RGB branch

The RGB branch is implemented in `src/models/rgb_stream.py` as `RGBStreamResNet`.

Key points:

- It uses a pretrained torchvision backbone.
- Supported backbones include `efficientnet_b4`, `convnext_tiny`, and `resnet50`.
- The default backbone is EfficientNet-B4.
- The classifier head is removed so the module returns feature vectors, not logits.
- The branch exposes `feature_dim` for downstream fusion.

The RGB stream is designed to be compatible with both direct feature extraction and wrapper checkpoints. The helper `normalize_rgb_checkpoint_state_dict` handles wrapper prefixes and older checkpoint layouts.

### 6.3 FFT branch

The FFT branch is implemented in `src/models/fft_stream.py` as `FFTStreamCNN`.

Core strategy:

- Convert input images to grayscale when needed.
- Compute an FFT spectrum with either `fft2` or `rfft2`.
- Apply log magnitude scaling.
- Normalize each spectrum to `[0, 1]`.
- Feed the spectrum through a lightweight CNN.

The FFT model returns a 256-dimensional embedding. It is intentionally compact so it can be trained and deployed cheaply compared with a second large RGB backbone.

The module also contains:

- `compute_fft_magnitude` for preprocessing.
- `FFTBlock` residual building blocks.
- `FFTOnlyClassifier` for ablation and standalone experiments.
- `normalize_fft_checkpoint_state_dict` for compatibility with older checkpoints and Kaggle-style wrappers.

### 6.4 Fusion head

`src/models/fusion_model.py` defines `FusionModel`.

Behavior:

- It receives an RGB backbone and an FFT backbone.
- It extracts features from both streams.
- It concatenates the embeddings.
- It passes the fused vector through a small MLP classifier.

Default fusion MLP shape:

- Input: RGB feature dimension + FFT feature dimension.
- Hidden layers: `256 -> 64`.
- Output: 2 classes.

The model can freeze backbones by default, which is useful when training only the fusion head on top of pretrained encoders.

---

## 7. Preprocessing Strategy

### 7.1 Face detection

The visual pipeline assumes face-centric inference. `src/data/face_detector.py` implements a two-level detection strategy:

1. MTCNN from `facenet-pytorch` when available.
2. OpenCV Haar cascade fallback when MTCNN is unavailable.

This is a pragmatic reliability choice. If the neural face detector is missing from the environment, the application still functions using a classical detector.

The detector exposes:

- Bounding-box detection.
- Cropping and resizing to 224x224.
- Path-based loading for image files.

### 7.2 Image augmentation

`src/data/augmentation.py` defines the training and evaluation transforms.

Training augmentation includes:

- Resize to 224x224.
- Horizontal flip.
- Rotation.
- Image compression noise.
- Brightness/contrast perturbation.
- ImageNet normalization.

Evaluation uses only resize plus normalization to keep validation deterministic.

### 7.3 FFT preprocessing in the data pipeline

`src/utils/fft_transform.py` re-exports the FFT preprocessing helper so dataset code can use the frequency transform without importing the full model module directly.

### 7.4 Dataset loading

`src/data/dataset.py` implements `DeepfakeDataset`.

Key features:

- CSV-driven input with required columns `path`, `label`, and `source_dataset`.
- Optional root directory resolution.
- Optional FFT tensor generation.
- Optional metadata return.
- Path validation at initialization.
- Recursive fallback when an image file is missing at runtime.

The dataset returns either:

- `(rgb_tensor, label)` when FFT is disabled.
- `(rgb_tensor, fft_tensor, label)` when FFT is enabled.
- Metadata can be appended for debugging or auditing.

---

## 8. Training Strategy

The project shows two distinct training flows.

### 8.1 Fusion training

`scripts/train_fusion.py` is the main training pipeline for the dual-stream model.

This script:

- Loads train and validation CSV splits.
- Builds train and eval dataloaders.
- Instantiates RGB and FFT backbones.
- Loads optional pretrained checkpoints.
- Builds the fusion classifier.
- Uses class-weighted cross entropy.
- Applies StepLR scheduling.
- Supports early stopping.
- Saves both `fusion_last.pt` and `fusion_best.pt`.
- Writes a JSON training history file.

Important strategy choices:

- Backbones can be frozen or unfrozen.
- FFT preprocessing can be forced to `fft2` or `rfft2`.
- The script detects and normalizes checkpoint formats from prior training runs.

### 8.2 FFT-only training

`src/training/train.py` trains an `FFTOnlyClassifier` baseline.

This path is useful for ablation and validation of the FFT branch in isolation.

Its strategy includes:

- Dataset loading with RGB input.
- FFT computed inside the model forward pass.
- Weighted loss to address class imbalance.
- AdamW optimization.
- OneCycleLR scheduling.
- Gradient clipping.
- Optional threshold calibration after training.

The script computes validation metrics and can persist calibrated thresholds into a sidecar JSON file.

### 8.3 Class imbalance handling

The dataset is not balanced. The project handles this in multiple ways:

- Class-weighted loss in training.
- Threshold calibration.
- F1 and ROC-AUC tracking.
- Explicit metric reporting instead of accuracy-only evaluation.

This is a good strategy for binary classification in a skewed dataset because raw accuracy can hide weak minority-class recall.

---

## 9. Inference Pipeline

The core inference layer is in `src/api/utils/inference.py` through `FusionInferenceService`.

### 9.1 Model loading

`load_model()` supports several modes:

- `fusion`
- `rgb`
- `fft`
- `audio`

It also handles legacy checkpoint formats by:

- Stripping `module.` prefixes from distributed-training checkpoints.
- Detecting nested dictionaries such as `state_dict` or `model_state_dict`.
- Heuristically remapping older unnamed wrappers.

### 9.2 Temperature calibration

If `results/checkpoints/temperature.json` exists, the service loads a scalar temperature for probability scaling.

This is a lightweight calibration mechanism that improves probability stability without changing the base network.

### 9.3 Image inference flow

For images, the service:

- Converts BGR to RGB.
- Applies evaluation transforms.
- Detects faces.
- Crops faces to 224x224.
- Runs the fusion model per face.
- Aggregates probabilities.
- Optionally generates Grad-CAM heatmaps.

### 9.4 Video inference flow

For videos, the service:

- Samples frames with a stride.
- Limits the number of processed frames.
- Runs per-frame image inference.
- Aggregates fake probabilities over sampled frames.
- Optionally performs audio inference if an audio model is loaded.

This gives the system a pragmatic video strategy: analyze a limited number of representative frames instead of decoding every frame in large videos.

### 9.5 Audio path

If an audio model is available, the service can combine visual and audio signals.

The combined score is controlled by environment variables:

- `AV_WEIGHT_VISUAL`
- `AV_WEIGHT_AUDIO`

That makes the final score interpretable and tunable without code changes.

---

## 10. API Design

The Flask app registers the following blueprints:

- `health_bp`
- `detect_bp`
- `fake_news_bp`
- `claim_verify_bp`

### 10.1 Health and info endpoints

- `GET /api/health` returns `{"status": "ok"}`.
- `GET /api/info` returns the model family, architecture summary, dataset name, and reported test metrics.

The info endpoint reports:

- Test accuracy: 0.9587
- Test AUC-ROC: 0.9951
- Test F1: 0.9690
- Supported inputs: JPEG, PNG, MP4

### 10.2 Deepfake detection endpoint

- `POST /api/detect` accepts file uploads.
- It supports images and videos.
- It validates file type and size before inference.

Image responses include:

- media type
- original dimensions
- elapsed time
- face-level results

Video responses include:

- number of analyzed frames
- fake frame count
- frame-level timeline data
- optional audio result
- optional visual/audio combined result

### 10.3 Fake news endpoint

- `POST /api/detect-text`
- `POST /api/detect-text/batch`

These endpoints accept text bodies and return fake/real verdicts with probabilities, confidence, and timing.

### 10.4 Claim verification endpoint

- `POST /api/claim-verify`
- `POST /api/claim-verify/batch`

These endpoints verify factual claims with support/refute/unknown probabilities.

### 10.5 Multi-agent research endpoint

- `POST /api/research-claim`

This is the longer-running research workflow backed by the multi-agent system. It returns a synthesized report and the source list used to support it.

### 10.6 Input validation strategy

The API enforces:

- Allowed file extensions.
- Maximum image and video sizes.
- Minimum and maximum character lengths for text and claim requests.
- Bounded batch sizes.

This is important because the app accepts untrusted user input and dispatches it into ML pipelines.

---

## 11. Frontend and UX

The main UI lives in `src/api/templates/index.html`, `src/api/static/app.js`, and `src/api/static/style.css`.

### 11.1 Visual design

The interface uses:

- A branded DeepGuard header.
- A three-tab layout.
- A clean light theme.
- Card-based layout for input and results.
- Color-coded verdict states.

### 11.2 Visual detection tab

The visual tab supports:

- Drag and drop uploads.
- File browsing.
- Image preview with face overlays.
- Video playback.
- Real/fake probability bar.
- Frame timeline for video analysis.
- Optional Grad-CAM heatmaps for suspicious faces.

### 11.3 Fake news tab

The fake-news tab supports:

- Multi-line text input.
- Character counting.
- Binary verdict display.
- Probability bar for real vs fake.
- Timing and summary statistics.

### 11.4 Claim verification tab

The claim tab supports:

- Claim input.
- Character counting.
- Three-way verdict display: supported, refuted, not enough info.
- Probability bar with support/refute/neutral distribution.

### 11.5 Frontend behavior

The JavaScript controller handles:

- Tab switching.
- Upload interactions.
- Drag-and-drop.
- Fetch-based API calls.
- Result rendering.
- Face overlay drawing.
- Frame timeline creation.
- Error states and resets.

The styling is intentionally compact and functional rather than decorative.

---

## 12. Configuration and Environment

### 12.1 `configs/default.yaml`

The default config sets:

- Data paths.
- Image size.
- Frame stride.
- Max frames per video.
- Minimum face size.
- Feature dimensions.
- Fusion hidden layers.
- Dropout values.
- Training hyperparameters.
- Augmentation policy.
- Output paths.

### 12.2 Environment variables

The runtime reads several environment variables:

- `CHECKPOINT`
- `DEVICE`
- `THRESHOLD`
- `MODE`
- `FAKE_NEWS_CHECKPOINT`
- `FAKE_NEWS_DEVICE`
- `FAKE_NEWS_USE_HF_INFERENCE`
- `FAKE_NEWS_PROVIDER`
- `FAKE_NEWS_POS_LABEL`
- `FAKE_NEWS_NEG_LABEL`
- `CLAIM_VERIFY_CHECKPOINT`
- `CLAIM_VERIFY_DEVICE`
- `CLAIM_VERIFY_USE_HF_INFERENCE`
- `CLAIM_VERIFY_PROVIDER`
- `CLAIM_VERIFY_SUPPORT_LABEL`
- `CLAIM_VERIFY_REFUTE_LABEL`
- `CLAIM_VERIFY_NEUTRAL_LABEL`
- `HF_TOKEN`
- `HUGGINGFACEHUB_API_TOKEN`
- `OLLAMA_BASE_URL`
- `CLAIM_MAX_ITERATIONS`
- `AV_WEIGHT_VISUAL`
- `AV_WEIGHT_AUDIO`

This makes the application adaptable across local machines, notebooks, and deployment environments.

---

## 13. Checkpoints and Compatibility

The project has a fairly mature checkpoint compatibility layer.

### 13.1 RGB checkpoints

RGB checkpoints are normalized to handle:

- Wrapped classifiers.
- `module.` prefixes.
- Headless backbone loading.

### 13.2 FFT checkpoints

FFT checkpoints are normalized to handle:

- Backbone-prefixed checkpoints.
- Wrapper checkpoints containing classifier heads.
- Detection of whether the checkpoint was trained with `rfft2` preprocessing.

### 13.3 Fusion checkpoints

Fusion training checkpoints store:

- Epoch number.
- Model state dict.
- Optimizer state dict.
- Scheduler state dict.
- Metrics.

This means the project can resume training and also load older artifacts without forcing a single strict serialization format.

---

## 14. Metrics and Evaluation

The project uses standard binary metrics:

- Accuracy
- Precision
- Recall
- F1 score
- False positive rate
- ROC-AUC

The metrics helpers are centralized in `src/utils/metrics.py`, which implements:

- `probabilities_from_logits`
- `positive_class_probabilities`
- `roc_auc_score_binary`
- `binary_classification_metrics`

This is a good separation because it keeps metric logic reusable across training, evaluation, and inference.

### 14.1 Reported performance

The app exposes a reported test set summary in `/api/info`:

- Accuracy: 95.87%
- AUC-ROC: 0.9951
- F1: 0.9690

### 14.2 Threshold calibration

The project explicitly calibrates or tunes decision thresholds in some training scripts. That is important for a binary detector because the operating point can be adjusted depending on the product priority:

- Favor recall to catch more fakes.
- Favor precision to reduce false alarms.

---

## 15. External Repo Integration 1: Fake Content Detection

The separate repository `yosrkharrat/fake_content_detection` is a fake-news analysis system built around text and URL analysis. It contributes the product idea and the text-analysis direction of this workspace.

### 15.1 What that repo provides

Based on the repository structure and documentation, it includes:

- A FastAPI backend.
- A content scraper for URL analysis.
- A feature extractor for linguistic, sentiment, structural, and source credibility signals.
- A transformer-based fake-news model.
- An explainer that turns model output into readable factors and recommendations.
- A web interface.

### 15.2 Tech stack of that repo

The repo uses:

- Python.
- FastAPI.
- PyTorch.
- Hugging Face Transformers.
- scikit-learn.
- NLTK.
- TextBlob and optional spaCy.
- newspaper3k and BeautifulSoup for scraping.
- SHAP and LIME for explainability.

### 15.3 Strategy used in that repo

Its approach is not image-based. It works by:

- Scraping or ingesting text.
- Extracting 40+ features.
- Running a transformer classifier plus feature-based scoring.
- Generating a structured explanation with positive and negative factors.

This is relevant to the current project because it shows the same product theme from the textual misinformation angle. In the current workspace, the corresponding local seam is `src/models/fake_news_model.py` and the `POST /api/detect-text` endpoint.

### 15.4 How it fits this project

The current project reuses the idea of a dedicated fake-news analysis tab and a model wrapper that can load a local checkpoint or a Hugging Face-style inference endpoint. The UI and API in this workspace are therefore aligned with the external repo’s design philosophy, even though the implementation is now localized to this project.

---

## 16. External Repo Integration 2: Multi-Agent Research System

The separate repository `yosrkharrat/multi-agent-research-system` provides the research-oriented claim verification workflow.

### 16.1 What that repo provides

It is a local multi-agent orchestration system built with:

- Ollama.
- LangGraph.
- LangChain.
- Planner, researcher, critic, and writer agents.
- A supervisor router that prevents infinite critic/researcher loops.
- A FastAPI web interface.

### 16.2 Workflow strategy

The agent chain is:

1. Planner breaks a topic into 3 to 5 research questions.
2. Researcher uses a ReAct-style loop with search tools such as DuckDuckGo and Wikipedia.
3. Critic judges whether the findings are sufficiently grounded and specific.
4. Writer synthesizes the approved findings into a markdown report.
5. Supervisor enforces iteration caps and routes the loop.

### 16.3 Why this matters here

This repo is used in the current project as the research backbone for claim verification. The local code in `src/agents/claim_verifier.py` bootstraps the external agent system and exposes it as a simple `verify(claim)` interface.

### 16.4 Local integration details

The current project wraps that system rather than reimplementing it. The local wrapper:

- Dynamically loads the external agent modules.
- Avoids circular import problems.
- Constructs the graph using a `PipelineConfig`.
- Runs the graph on a claim string.
- Returns a normalized report with summary and sources.

### 16.5 Operational dependency

This workflow requires Ollama to be available. The external project documents that the user must run Ollama and pull the appropriate models before the agent system can function. In this workspace, the claim research endpoint returns a 503 if the agent is unavailable.

---

## 17. Audio and Multimodal Extension Points

The repository is not limited to RGB, FFT, and text. It also contains an audio path that can extend video analysis.

### 17.1 Audio models

`src/models/audio_stream.py` defines two approaches:

- A Wav2Vec2-based sequence classifier.
- A compact MFCC CNN.

### 17.2 Audio inference utilities

The API layer includes utilities to:

- Extract audio from a video.
- Run audio prediction.
- Combine audio and visual scores with weighted averaging.

### 17.3 Why this matters

Audio can carry deception cues that visual frames do not show. This makes the application more extensible and better suited for full media analysis, not only face manipulation detection.

---

## 18. Explainability and Debugging

The project includes explainability hooks and operational diagnostics:

- Grad-CAM visualizations for suspicious faces.
- Per-frame timelines for video analysis.
- Model temperature calibration.
- Validation and fallback logic for file types and media sizes.
- Checkpoint normalization helpers.
- Health and info endpoints.

These features are important because the system is intended for user-facing credibility judgments, not just internal benchmarking.

---

## 19. Deployment and Runtime Notes

### 19.1 Entry points

- `python run.py` launches the Flask app.
- The app listens on `0.0.0.0:5000` by default.

### 19.2 Production considerations

The codebase already contains a number of deployment-conscious choices:

- Request size caps.
- Explicit environment-variable control.
- Fast fail behavior when checkpoints are missing.
- Optional model loading instead of hard failing on every subsystem.

### 19.3 Common operational dependencies

- OpenCV must be installed for media inference.
- Transformers are needed for text and some audio paths.
- facenet-pytorch improves face detection but is optional.
- Ollama is needed for the multi-agent research workflow.

---

## 20. Strengths of the Design

This project is strong because it does not try to solve everything with a single model. Instead, it chooses the right mechanism for the right modality:

- Spatial CNN features for visible manipulations.
- Frequency-domain features for generative artifacts.
- Transformer text classification for misinformation.
- Multi-agent web research for longer-form claim verification.

Other strengths:

- Clear modularization.
- Good checkpoint compatibility.
- Practical UI and API integration.
- Support for image, video, text, and claim workflows.
- Optional multimodal extension via audio.

---

## 21. Limitations and Risks

The implementation is solid, but several limitations remain:

- The deepfake system still depends heavily on face detection quality.
- Video analysis samples only a limited number of frames.
- Text fake-news detection depends on the quality and domain fit of the loaded checkpoint.
- The multi-agent claim verification workflow depends on local Ollama availability and can be slow.
- Some documentation files in the repo are placeholders, so the source code is the authoritative reference.

These are not defects so much as realistic trade-offs of the chosen architecture.

---

## 22. Recommended Future Work

If this project continues, the highest-value next steps would be:

- Add a single unified report page that merges visual, textual, and claim results.
- Persist inference results to a database for auditability.
- Add background jobs for long-running video and research tasks.
- Extend the text model with explicit explanation outputs.
- Improve calibration and threshold selection per dataset split.
- Add a stronger evaluation harness for the claim research workflow.
- Add automated end-to-end tests for the Flask routes and frontend.

---

## 23. File-Level Summary

Most important files in this workspace:

- `run.py` starts the service.
- `src/api/app.py` wires all blueprints and model loaders.
- `src/api/templates/index.html` defines the UI.
- `src/api/static/app.js` implements frontend behavior.
- `src/api/static/style.css` implements the UI theme.
- `src/models/rgb_stream.py` defines the RGB backbone.
- `src/models/fft_stream.py` defines the FFT backbone and ablations.
- `src/models/fusion_model.py` defines the fusion classifier.
- `src/data/dataset.py` loads face crops and FFT tensors.
- `src/data/face_detector.py` finds faces.
- `src/training/train.py` trains FFT-only baselines.
- `scripts/train_fusion.py` trains the main fusion model.
- `src/models/fake_news_model.py` wraps text fake-news inference.
- `src/models/claim_verify_model.py` wraps NLI-style claim verification.
- `src/agents/claim_verifier.py` bridges to the external multi-agent repo.

---

## 24. Final Assessment

This is a genuinely multi-layered credibility detection system. The deepfake detector is the technical center of gravity, but the surrounding application is what turns it into a product: a clean UI, multiple analysis modes, calibration and fallback logic, and external intelligence integrations for text misinformation and claim research.

If you want a single sentence summary, it is this:

**The project combines a dual-stream deepfake detector, a text-based fake-news classifier, and a multi-agent claim research system into one Flask-powered credibility analysis platform.**
