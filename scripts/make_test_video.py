"""Create a short test MP4 from existing face crop JPGs.

Usage:
    python scripts/make_test_video.py --label real --out test_real.mp4
    python scripts/make_test_video.py --label fake --out test_fake.mp4
"""

import argparse
import glob
import os
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["real", "fake"], default="real")
    parser.add_argument("--out", default="test_video.mp4")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to include")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    base = f"data/raw/FaceForensics++_C23/{args.label}"
    pattern = os.path.join(base, "**", "*.jpg")
    jpgs = sorted(glob.glob(pattern, recursive=True))[: args.frames]

    if not jpgs:
        print(f"No JPGs found under {base}")
        return

    sample = cv2.imread(jpgs[0])
    h, w = sample.shape[:2]

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for path in jpgs:
        frame = cv2.imread(path)
        if frame is not None:
            writer.write(frame)
    writer.release()

    print(f"Saved {len(jpgs)} frames → {args.out}  ({w}x{h} @ {args.fps} fps)")

if __name__ == "__main__":
    main()
