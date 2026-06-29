"""
Video Cropping Tool - OpenCV Version
Version: 1.4.0
Description: Extracts sub-clips from a video based on h:m:s timestamps.
"""

import cv2
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.paths import CLIPS_OUTPUT_DIR, DEFAULT_SOURCE, resolve_path

# =============================================================================
# TIME FORMATTING GUIDE:
# Use strings in the following formats for start_time and end_time:
# - "Seconds"          -> "30"      (30 seconds)
# - "Minutes:Seconds"  -> "1:20"    (1 minute 20 seconds)
# - "Hours:Min:Sec"    -> "1:15:30" (1 hour 15 minutes 30 seconds)
# =============================================================================


def time_to_seconds(t_str):
    """Converts h:m:s, m:s, or s string to total integer seconds."""
    if not t_str:
        return None
    parts = list(map(int, t_str.split(":")))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 1:
        return parts[0]
    return 0


def format_time_for_name(t_str):
    """Converts colon-based time (1:20:05) to filename-safe string (1h20m5s)."""
    if not t_str:
        return "0s"

    clean = t_str.replace(":", "m") + "s"
    if t_str.count(":") == 2:
        clean = clean.replace("m", "h", 1)
    return clean


def sanitize_name(name):
    """Removes spaces and special characters from the filename."""
    name_only = Path(name).stem
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name_only)
    return re.sub(r"_+", "_", sanitized).strip("_")


def trim_video_with_naming(input_path, start_time=None, end_time=None, save_frames=False):
    output_folder = CLIPS_OUTPUT_DIR
    output_folder.mkdir(exist_ok=True)

    input_resolved = resolve_path(input_path)
    base_name = sanitize_name(input_resolved)

    st_label = format_time_for_name(start_time)
    en_label = format_time_for_name(end_time) if end_time else "end"

    output_filename = f"{base_name}_st{st_label}_en{en_label}.mp4"
    output_path = output_folder / output_filename

    if save_frames:
        frames_folder = output_folder / "frames" / output_filename[:-4]
        os.makedirs(frames_folder, exist_ok=True)

    cap = cv2.VideoCapture(str(input_resolved))
    if not cap.isOpened():
        print(f"Error: Could not open {input_resolved}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_sec = time_to_seconds(start_time) if start_time else 0
    end_sec = time_to_seconds(end_time) if end_time else (total_frames / fps)

    start_frame = int(round(start_sec * fps))
    end_frame = math.ceil(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("--- Video clipping v1.4.0 ---")
    print(f"Source: {input_resolved.name}")
    print(f"Target: {output_filename}")
    print(f"FPS: {fps}")
    print(f" Duration: {end_sec - start_sec}")

    frame_count = start_frame
    while cap.isOpened() and frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

        if save_frames:
            frame_img_name = (
                f"{base_name}_st{st_label}_en{en_label}_frame_{frame_count}.png"
            )
            cv2.imwrite(str(frames_folder / frame_img_name), frame)

        if frame_count % 100 == 0:
            print(f"frame:{frame_count} completed")
        frame_count += 1

    cap.release()
    out.release()
    print(f"Done! Saved {frame_count - start_frame} frames to {output_path}")


if __name__ == "__main__":
    video_input = str(DEFAULT_SOURCE)
    trim_video_with_naming(
        video_input, start_time="00:00", end_time="00:50", save_frames=False
    )
