#!/usr/bin/env python3
"""Run tracking experiments from configs/experiments/all_exps.yaml."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.paths import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    PROJECT_ROOT,
    ensure_outputs_dir,
    resolve_path,
)

DEFAULT_ALL_EXPS = PROJECT_ROOT / "configs" / "experiments" / "all_exps.yaml"
TRACK_SCRIPT = PROJECT_ROOT / "scripts" / "track_players.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BoT-SORT tracking experiments from all_exps YAML."
    )
    parser.add_argument(
        "--all-exps",
        default=str(DEFAULT_ALL_EXPS),
        help="All experiments YAML (default: configs/experiments/all_exps.yaml)",
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
        "--reid-model",
        default="auto",
        help='ReID encoder: "auto" or path/name e.g. yolo26n-reid.onnx',
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root folder for experiment outputs (default: outputs)",
    )
    parser.add_argument(
        "--result-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep result video after each experiment (default: on)",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Run only these experiment ids (default: all in all_exps)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.35,
        help="NMS IoU threshold passed to track_players (default: 0.35)",
    )
    return parser.parse_args()


def load_all_exps(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    experiments = data.get("experiments")
    if not experiments:
        raise ValueError(f"No experiments found in all_exps: {path}")
    return experiments


def load_tracker_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def relocate_frames(source_dir: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        return 0
    moved = 0
    for frame_path in sorted(source_dir.glob("*.jpg")):
        shutil.move(str(frame_path), str(dest_dir / frame_path.name))
        moved += 1
    if source_dir.exists() and not any(source_dir.iterdir()):
        source_dir.rmdir()
    return moved


def run_single_experiment(
    exp: dict[str, Any],
    *,
    source: Path,
    model: Path,
    reid_model: str,
    output_root: Path,
    result_video: bool,
    iou: float,
) -> dict[str, Any]:
    exp_id = exp["id"]
    exp_name = exp.get("name", exp_id)
    tracker_yaml = resolve_path(exp["tracker"])
    conf = float(exp["conf"])
    started_at = datetime.now(timezone.utc).isoformat()

    exp_dir = ensure_outputs_dir(output_root / exp_id)
    images_dir = exp_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    video_stem = source.stem
    details_path = exp_dir / f"{video_stem}_details.json"
    video_path = exp_dir / f"{video_stem}_tracked.mp4"
    meta_path = exp_dir / "experiment_meta.json"
    frames_src = PROJECT_ROOT / "frame_saved" / video_stem

    cmd = [
        sys.executable,
        str(TRACK_SCRIPT),
        "--source",
        str(source),
        "--model",
        str(model),
        "--tracker",
        str(tracker_yaml),
        "--reid-model",
        reid_model,
        "--output-dir",
        str(exp_dir),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--save-frame",
        "--no-show-conf",
        "--no-show-coords",
    ]

    print(f"\n{'=' * 60}")
    print(f"Experiment {exp_id} ({exp_name})")
    print(f"Tracker: {tracker_yaml.name}  conf={conf}")
    print(f"Output:  {exp_dir}")
    print(f"{'=' * 60}")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    finished_at = datetime.now(timezone.utc).isoformat()

    status = "success" if proc.returncode == 0 else "failed"
    error = None if proc.returncode == 0 else f"track_players exited with code {proc.returncode}"

    frames_moved = 0
    video_kept = False

    if status == "success":
        frames_moved = relocate_frames(frames_src, images_dir)
        if result_video and video_path.exists():
            video_kept = True
        elif video_path.exists():
            video_path.unlink()

    tracker_config = load_tracker_config(tracker_yaml)
    meta = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "status": status,
        "error": error,
        "started_at": started_at,
        "finished_at": finished_at,
        "source": str(source),
        "video_stem": video_stem,
        "model": str(model),
        "tracker_yaml": str(tracker_yaml),
        "tracker_config": tracker_config,
        "reid_model": reid_model,
        "conf": conf,
        "iou": iou,
        "result_video": result_video,
        "paths": {
            "experiment_dir": str(exp_dir),
            "details_json": str(details_path) if details_path.exists() else None,
            "video": str(video_path) if video_kept else None,
            "images_dir": str(images_dir),
        },
        "frames_saved": frames_moved,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Status: {status}")
    if status == "success":
        print(f"Details: {details_path.name}" + (" (ok)" if details_path.exists() else " (missing)"))
        print(f"Images:  {frames_moved} frames -> {images_dir}")
        print(f"Video:   {'kept' if video_kept else 'skipped/deleted'}")
    else:
        print(f"Error:   {error}")

    return meta


def write_summary(output_root: Path, results: list[dict[str, Any]]) -> Path:
    summary_path = output_root / "experiments_summary.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": results,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return summary_path


def main() -> None:
    args = parse_args()

    all_exps_path = resolve_path(args.all_exps)
    source = resolve_path(args.source)
    model = resolve_path(args.model)
    output_root = ensure_outputs_dir(args.output_root)

    if not all_exps_path.exists():
        raise FileNotFoundError(f"all_exps not found: {all_exps_path}")
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")
    if not TRACK_SCRIPT.exists():
        raise FileNotFoundError(f"Track script not found: {TRACK_SCRIPT}")

    experiments = load_all_exps(all_exps_path)
    if args.experiments:
        selected = set(args.experiments)
        experiments = [e for e in experiments if e["id"] in selected]
        missing = selected - {e["id"] for e in experiments}
        if missing:
            raise ValueError(f"Unknown experiment ids: {sorted(missing)}")

    print("--- Tracker Experiment Runner ---")
    print(f"All exps: {all_exps_path}")
    print(f"Source:   {source.name}")
    print(f"Runs:     {len(experiments)} experiment(s)")
    print(f"Video:    {'keep' if args.result_video else 'discard after run'}")

    results: list[dict[str, Any]] = []
    for exp in experiments:
        try:
            meta = run_single_experiment(
                exp,
                source=source,
                model=model,
                reid_model=args.reid_model,
                output_root=output_root,
                result_video=args.result_video,
                iou=args.iou,
            )
        except Exception as exc:
            meta = {
                "experiment_id": exp.get("id"),
                "experiment_name": exp.get("name"),
                "status": "failed",
                "error": str(exc),
            }
            print(f"Experiment {exp.get('id')} crashed: {exc}")
        results.append(meta)

    summary_path = write_summary(output_root, results)
    succeeded = sum(1 for r in results if r.get("status") == "success")
    print(f"\nDone. {succeeded}/{len(results)} succeeded.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
