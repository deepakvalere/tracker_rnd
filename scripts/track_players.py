#!/usr/bin/env python3
"""Detect and track players with YOLO + BoT-SORT ReID."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.detector import PlayerTracker
from scripts.utils.paths import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    DEFAULT_TRACKER,
    ensure_outputs_dir,
    resolve_path,
)
from scripts.utils.video_io import LiveViewer, create_writer, open_capture
from scripts.utils.visualizer import draw_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track players using YOLO detection and BoT-SORT with ReID."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Input video path (default: inputs/vid1.mp4)",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="YOLO player detection weights (default: models/player.pt)",
    )
    parser.add_argument(
        "--tracker",
        default=str(DEFAULT_TRACKER),
        help="BoT-SORT tracker config (default: configs/botsort_reid.yaml)",
    )
    parser.add_argument(
        "--reid-model",
        default="auto",
        help='ReID encoder: "auto" or path/name e.g. yolo26n-reid.onnx',
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder for annotated video (default: outputs)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window (ESC or q to stop)",
    )
    parser.add_argument(
        "--no-coords",
        action="store_true",
        help="Hide (cx,cy) text labels; center dot remains visible",
    )
    parser.add_argument(
        "--show-conf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show detection confidence on each bounding box (default: on)",
    )
    parser.add_argument(
        "--show-coords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show (cx,cy) text labels on each track (default: on)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.35,
        help="NMS IoU threshold (default: 0.35)",
    )
    parser.add_argument(
        "--save-frame",
        action="store_true",
        help="Save annotated frames to frame_saved/<video>/frameN.jpg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = resolve_path(args.source)
    model_path = resolve_path(args.model)
    tracker_yaml = resolve_path(args.tracker)
    output_dir = ensure_outputs_dir(args.output_dir)

    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not tracker_yaml.exists():
        raise FileNotFoundError(f"Tracker config not found: {tracker_yaml}")

    cap = open_capture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    output_path = output_dir / f"{source.stem}_tracked.mp4"
    writer = create_writer(output_path, fps, size)
    viewer = LiveViewer(enabled=args.show)

    tracker = PlayerTracker(
        model_path=model_path,
        tracker_yaml=tracker_yaml,
        reid_model=args.reid_model,
        conf=args.conf,
        iou=args.iou,
    )

    show_coords = args.show_coords and not args.no_coords
    show_conf = args.show_conf
    save_frame = args.save_frame
    frame_count = 0
    frames_dir = ROOT / "frame_saved" / source.stem
    if save_frame:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print("--- Player Tracking (YOLO + BoT-SORT ReID) ---")
    print(f"Source: {source.name}")
    print(f"Model:  {model_path.name}")
    print(f"Output: {output_path}")
    print(f"Live:   {'on (ESC/q to stop)' if args.show else 'off'}")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            tracks = tracker.track_frame(frame)
            annotated = draw_tracks(
                frame, tracks, show_coords=show_coords, show_conf=show_conf
            )
            frame_num = frame_count + 1
            cv2.putText(
                annotated,
                str(frame_num),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if save_frame:
                cv2.imwrite(str(frames_dir / f"frame{frame_num}.jpg"), annotated)
            writer.write(annotated)
            frame_count += 1

            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")

            if viewer.show(annotated):
                print("Stopped by user (ESC/q).")
                break
    finally:
        cap.release()
        writer.release()
        viewer.close()

    print(f"Done. Wrote {frame_count} frames to {output_path}")


if __name__ == "__main__":
    main()
