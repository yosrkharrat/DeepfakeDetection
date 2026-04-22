# Deepfake Detection

Dual-stream CNN (RGB + FFT) for deepfake detection, trained on FaceForensics++ C23.

## Project Structure

```
DeepfakeDetection/
├── data/
│   ├── raw/                  # Raw videos (not tracked by git)
│   │   └── FaceForensics++_C23/
│   │       ├── original/     # Real videos
│   │       ├── Deepfakes/
│   │       ├── Face2Face/
│   │       ├── FaceShifter/
│   │       ├── FaceSwap/
│   │       └── NeuralTextures/
│   ├── processed/            # Face crops (not tracked by git)
│   └── splits/               # Train/val/test CSVs (not tracked by git)
├── scripts/
│   ├── precompute_faces.py   # Step 1 – extract face crops from videos
│   └── make_splits.py        # Step 2 – generate stratified CSV splits
├── src/
│   ├── data/
│   │   ├── face_detector.py  # MTCNN / Haar cascade wrapper
│   │   ├── dataset.py        # PyTorch Dataset
│   │   └── augmentation.py
│   ├── models/
│   ├── training/
│   └── utils/
├── configs/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Pipeline

The pipeline runs in two steps. All output goes under `data/` which is git-ignored.

### Step 1 — Precompute face crops

Extracts 224×224 face crops from raw videos using MTCNN (falls back to OpenCV Haar cascade).

```bash
python scripts/precompute_faces.py \
    --raw-dir data/raw/FaceForensics++_C23 \
    --out-dir data/processed \
    --frame-stride 30 \
    --max-frames-per-video 10 \
    --workers 7 \
    --device cpu
```

| Argument | Default | Description |
|---|---|---|
| `--raw-dir` | `data/raw/FaceForensics++_C23` | Root of raw video dataset |
| `--out-dir` | `data/processed` | Output directory for crops |
| `--frame-stride` | `30` | Sample every Nth frame |
| `--max-frames-per-video` | `10` | Max frames sampled per video |
| `--workers` | `cpu_count - 1` | Parallel worker processes |
| `--device` | `cpu` | `cpu` or `cuda` for MTCNN |
| `--skip-existing` | `True` | Resume — skip already-processed videos |

Outputs:
- `data/processed/FaceForensics++_C23/{real,fake}/<video_id>/frame_*.jpg`
- `data/processed/precompute_manifest.csv`
- `data/processed/precompute_report.json`

#### Dataset stats (FaceForensics++ C23, 7,000 videos)

| | Count |
|---|---|
| Total face crops | 32,898 |
| Real (label 0) | 10,499 |
| Fake (label 1) | 22,399 |

> **Note:** The dataset is ~2:1 fake/real. Use a weighted loss or weighted sampler during training.

### Step 2 — Generate train/val/test splits

Stratified split by label (70 / 15 / 15).

```bash
python scripts/make_splits.py \
    --processed-dir data/processed \
    --output-dir data/splits \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

Outputs: `data/splits/train.csv`, `val.csv`, `test.csv` — each with columns `path`, `label`, `source_dataset`.

#### Split summary

| Split | Total | Real | Fake |
|---|---|---|---|
| Train | 23,028 | 7,349 | 15,679 |
| Val | 4,933 | 1,574 | 3,359 |
| Test | 4,937 | 1,576 | 3,361 |

## Face Detector

`src/data/face_detector.py` wraps two backends in priority order:

1. **MTCNN** (`facenet-pytorch`) — neural network detector, accurate
2. **OpenCV Haar cascade** — fallback if facenet-pytorch is not installed

Install MTCNN:
```bash
pip install facenet-pytorch
```
