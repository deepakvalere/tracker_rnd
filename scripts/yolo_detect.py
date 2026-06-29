#!/usr/bin/env python3
"""Run YOLO player detection on a video and save annotated frames."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.paths import PROJECT_ROOT, resolve_path
from scripts.utils.video_io import open_capture

video_path = r"C:\Users\ATM\Desktop\pshi\siq-labs-cv\inputs\6b9ce998-f28d-4f99-b49f-5218d271f4f4.MOV"
output_path = PROJECT_ROOT / "outputs" / "detected_frames"
model_path = PROJECT_ROOT / "models" / "player.pt"
conf = 0.25
use_tracker = True


def main() -> None:
    source = resolve_path(video_path)
    out_dir = resolve_path(output_path)
    weights = resolve_path(model_path)

    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")
    if not weights.exists():
        raise FileNotFoundError(f"Model not found: {weights}")

    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    cap = open_capture(source)

    frame_idx = 0
    print("--- YOLO Player Detection ---")
    print(f"Source: {source}")
    print(f"Model:  {weights.name}")
    print(f"Output: {out_dir}")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if use_tracker:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=conf,
                    verbose=False,
                )
            else:
                results = model(frame, conf=conf, verbose=False)
            annotated = results[0].plot()

            frame_file = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_file), annotated)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"Saved {frame_idx} frames...")
    finally:
        cap.release()

    print(f"Done. Saved {frame_idx} frames to {out_dir}")


if __name__ == "__main__":
    main()
