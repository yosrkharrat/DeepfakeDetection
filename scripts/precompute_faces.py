"""Batch precompute 224x224 face crops from raw videos/images.

Pipeline (v2):
1. Discover media files under data/raw
2. Sample video frames (or use images directly)
3. Detect faces (MTCNN if available, OpenCV Haar fallback)
4. Crop/resize faces to model input size
5. Save crops to data/processed and emit a manifest CSV

Parallelism: each worker process owns its own FaceDetector instance.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.data.face_detector import FaceDetector


VALID_MEDIA_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".webp"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CLASS_LABEL_MAP = {
    "original": 0,
    "deepfakes": 1,
    "face2face": 1,
    "faceswap": 1,
    "neuraltextures": 1,
    "faceshifter": 1,
    "deepfakedetection": 1,
}

# Per-worker globals (set by pool initializer)
_worker_detector: Optional[FaceDetector] = None
_worker_args: Optional[argparse.Namespace] = None


@dataclass
class PipelineStats:
    videos_total: int = 0
    media_processed: int = 0
    media_failed: int = 0
    frames_sampled: int = 0
    frames_no_face: int = 0
    faces_too_small: int = 0
    faces_saved: int = 0

    def merge(self, other: "PipelineStats") -> None:
        self.media_processed += other.media_processed
        self.media_failed += other.media_failed
        self.frames_sampled += other.frames_sampled
        self.frames_no_face += other.frames_no_face
        self.faces_too_small += other.faces_too_small
        self.faces_saved += other.faces_saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute face crops from raw media.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/FaceForensics++_C23"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest-path", type=Path, default=Path("data/processed/precompute_manifest.csv"))
    parser.add_argument("--dataset-name", default="FaceForensics++_C23")
    parser.add_argument("--frame-stride", type=int, default=30, help="Sample every Nth frame from videos.")
    parser.add_argument("--max-frames-per-video", type=int, default=10, help="Cap sampled frames per video.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-face-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Parallel worker processes.")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip videos whose output directory already has crops.")
    parser.add_argument("--report-path", type=Path, default=Path("data/processed/precompute_report.json"))
    return parser.parse_args()


def infer_label_and_class(media_path: Path, raw_root: Path) -> Tuple[Optional[int], Optional[str]]:
    rel_parts = [part.lower() for part in media_path.relative_to(raw_root).parts]
    for part in rel_parts:
        if part in CLASS_LABEL_MAP:
            label = CLASS_LABEL_MAP[part]
            return label, ("real" if label == 0 else "fake")
    return None, None


def discover_media(raw_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VALID_MEDIA_EXTENSIONS:
            continue
        label, class_name = infer_label_and_class(path, raw_root)
        if label is None or class_name is None:
            continue
        records.append({"path": path, "label": label, "class_name": class_name, "video_id": path.stem})
    records.sort(key=lambda r: str(r["path"]))
    return records


def sample_video_frames(video_path: Path, frame_stride: int, max_frames: int) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        frame_idx = -1
        emitted = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % frame_stride != 0:
                continue
            yield frame_idx, frame
            emitted += 1
            if emitted >= max_frames:
                break
    finally:
        cap.release()


def iter_media_frames(path: Path, frame_stride: int, max_frames_per_video: int) -> Iterator[Tuple[int, np.ndarray]]:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is not None:
            yield 0, frame
        return
    yield from sample_video_frames(path, frame_stride=frame_stride, max_frames=max_frames_per_video)


def detect_and_crop_faces(
    frame_bgr: np.ndarray,
    detector: FaceDetector,
    image_size: int,
    min_face_size: int,
) -> Tuple[List[np.ndarray], int]:
    crops: List[np.ndarray] = []
    too_small = 0
    for x, y, w, h in detector.detect_boxes(frame_bgr):
        if w < min_face_size or h < min_face_size:
            too_small += 1
            continue
        crop = frame_bgr[y: y + h, x: x + w]
        if crop.size == 0:
            continue
        crops.append(cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA))
    return crops, too_small


def save_face_crop(
    crop_bgr: np.ndarray,
    out_dir: Path,
    dataset_name: str,
    class_name: str,
    video_id: str,
    frame_idx: int,
    face_idx: int,
) -> Path:
    target_dir = out_dir / dataset_name / class_name / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"frame_{frame_idx:06d}_face_{face_idx:02d}.jpg"
    if not cv2.imwrite(str(out_path), crop_bgr):
        raise IOError(f"Failed to write face crop to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Worker initializer — runs once per process in the pool
# ---------------------------------------------------------------------------

def _worker_init(device: str, args_dict: dict) -> None:
    global _worker_detector, _worker_args
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    _worker_detector = FaceDetector(device=device)
    # Reconstruct a simple namespace so worker_process can use it
    _worker_args = argparse.Namespace(**args_dict)


# ---------------------------------------------------------------------------
# Worker task — processes a single media record, returns (rows, stats_dict, log)
# ---------------------------------------------------------------------------

def _worker_process(media_record: Dict[str, object]) -> Tuple[List[Dict], Dict, str]:
    assert _worker_detector is not None and _worker_args is not None
    args = _worker_args
    detector = _worker_detector

    path = media_record["path"]
    label = int(media_record["label"])
    class_name = str(media_record["class_name"])
    video_id = str(media_record["video_id"])

    stats = PipelineStats()
    manifest_rows: List[Dict[str, object]] = []
    any_face_saved = False

    try:
        for frame_idx, frame_bgr in iter_media_frames(path, args.frame_stride, args.max_frames_per_video):
            stats.frames_sampled += 1
            crops, too_small = detect_and_crop_faces(frame_bgr, detector, args.image_size, args.min_face_size)
            stats.faces_too_small += too_small
            if not crops:
                stats.frames_no_face += 1
                continue
            for face_idx, crop in enumerate(crops):
                saved = save_face_crop(crop, args.out_dir, args.dataset_name, class_name, video_id, frame_idx, face_idx)
                manifest_rows.append({"path": saved.as_posix(), "label": label, "source_dataset": args.dataset_name})
                stats.faces_saved += 1
                any_face_saved = True

        if any_face_saved:
            stats.media_processed += 1
        else:
            stats.media_failed += 1
        log = ""
    except Exception as exc:
        stats.media_failed += 1
        log = f"[WARN] Failed {path}: {exc}"

    return manifest_rows, asdict(stats), log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_manifest(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source_dataset"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(stats: PipelineStats, report_path: Path, detector_backend: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(stats)
    payload["detector_backend"] = detector_backend
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if cv2 is None:
        print("[ERROR] OpenCV not installed. Run: pip install opencv-python")
        return 3
    if not args.raw_dir.exists():
        print(f"[ERROR] Raw directory not found: {args.raw_dir}")
        return 1

    # Probe detector backend in main process (for logging only)
    probe = FaceDetector(device=args.device)
    if probe.backend == "none":
        print("[ERROR] No face detector available.")
        return 2
    backend = probe.backend
    del probe

    media = discover_media(args.raw_dir)

    if args.skip_existing:
        def _has_crops(record: Dict[str, object]) -> bool:
            d = args.out_dir / args.dataset_name / str(record["class_name"]) / str(record["video_id"])
            return d.exists() and any(d.iterdir())
        before = len(media)
        media = [r for r in media if not _has_crops(r)]
        print(f"[INFO] Skipping {before - len(media)} already-processed videos")

    total = len(media)

    print(f"[INFO] Detector backend : {backend}")
    print(f"[INFO] Media to process  : {total}")
    print(f"[INFO] Workers           : {args.workers}")
    print(f"[INFO] Frames/video      : {args.max_frames_per_video}  (stride={args.frame_stride})")

    # Serialise Path objects so they survive pickling on Windows
    args_dict = vars(args).copy()
    args_dict["raw_dir"] = args.raw_dir
    args_dict["out_dir"] = args.out_dir
    args_dict["manifest_path"] = args.manifest_path
    args_dict["report_path"] = args.report_path

    stats = PipelineStats(videos_total=len(discover_media(args.raw_dir)))
    all_rows: List[Dict[str, object]] = []

    TASK_TIMEOUT = 60  # seconds per video before skipping

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.device, args_dict),
    ) as pool:
        futures = [(record, pool.apply_async(_worker_process, (record,))) for record in media]
        for idx, (record, future) in enumerate(futures, start=1):
            try:
                rows, s_dict, log = future.get(timeout=TASK_TIMEOUT)
                all_rows.extend(rows)
                stats.merge(PipelineStats(**s_dict))
                if log:
                    print(log, flush=True)
            except mp.TimeoutError:
                stats.media_failed += 1
                print(f"[WARN] Timeout skipping: {record['path']}", flush=True)
            except Exception as exc:
                stats.media_failed += 1
                print(f"[WARN] Error skipping {record['path']}: {exc}", flush=True)
            if idx % 100 == 0 or idx == total:
                print(f"[INFO] ({idx}/{total}) faces_saved={stats.faces_saved}", flush=True)

    write_manifest(all_rows, args.manifest_path)
    write_report(stats, args.report_path, detector_backend=backend)

    print("\n=== Precompute Summary ===")
    print(f"Media discovered      : {stats.videos_total}")
    print(f"Media processed       : {stats.media_processed}")
    print(f"Media failed/no faces : {stats.media_failed}")
    print(f"Frames sampled        : {stats.frames_sampled}")
    print(f"Frames with no face   : {stats.frames_no_face}")
    print(f"Faces too small       : {stats.faces_too_small}")
    print(f"Faces saved           : {stats.faces_saved}")
    print(f"Manifest              : {args.manifest_path}")
    print(f"Report                : {args.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
