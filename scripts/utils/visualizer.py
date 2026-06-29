from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette: list[tuple[int, int, int]] = [
        (180, 0, 0),
        (0, 140, 0),
        (0, 0, 180),
        (140, 0, 140),
        (0, 100, 140),
        (0, 70, 140),
        (100, 100, 0),
        (60, 0, 120),
        (0, 90, 90),
        (90, 90, 0),
    ]
    return palette[track_id % len(palette)]


def draw_tracks(
    frame: np.ndarray,
    tracks: list[dict[str, Any]],
    *,
    show_coords: bool = True,
    show_conf: bool = True,
) -> np.ndarray:
    annotated = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track["xyxy"]]
        track_id = track["track_id"]
        color = _track_color(track_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track_id:02d}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 8, label_size[1] + 4)
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        conf_text = f"{track['conf']:.2f}"
        conf_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        conf_y = max(y1 - 8, conf_size[1] + 4)
        if show_conf:
            cv2.putText(
                annotated,
                conf_text,
                (x2 - conf_size[0], conf_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 4, color, -1)

        if show_coords:
            coord_text = f"({cx},{cy})"
            cv2.putText(
                annotated,
                coord_text,
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return annotated
