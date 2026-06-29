from __future__ import annotations

from pathlib import Path

import cv2


def open_capture(source: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    return cap


def create_writer(
    output_path: str | Path, fps: float, size: tuple[int, int]
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    return writer


class LiveViewer:
    """Optional live preview with WINDOW_NORMAL and ESC/q quit support."""

    WINDOW_NAME = "Player Tracking"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        if self.enabled:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

    def show(self, frame) -> bool:
        """Show frame. Returns True if user requested quit (ESC or q)."""
        if not self.enabled:
            return False

        cv2.imshow(self.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        return check_quit(key)

    def close(self) -> None:
        if self.enabled:
            cv2.destroyWindow(self.WINDOW_NAME)


def check_quit(key: int) -> bool:
    return key in (27, ord("q"), ord("Q"))
