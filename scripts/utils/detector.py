from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from scripts.utils.paths import PROJECT_ROOT


class PlayerTracker:
    """YOLO detection + BoT-SORT ReID tracking wrapper."""

    def __init__(
        self,
        model_path: str | Path,
        tracker_yaml: str | Path,
        reid_model: str = "auto",
        conf: float = 0.25,
        iou: float = 0.35,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.tracker_config = self._prepare_tracker_config(
            Path(tracker_yaml), reid_model
        )
        self.model = YOLO(str(model_path))

    def _prepare_tracker_config(
        self, tracker_yaml: Path, reid_model: str
    ) -> str:
        with tracker_yaml.open(encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        cfg["with_reid"] = True
        if reid_model:
            cfg["model"] = reid_model

        runtime_cfg = PROJECT_ROOT / "configs" / ".botsort_reid_runtime.yaml"
        with runtime_cfg.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        return str(runtime_cfg)

    def track_frame(self, frame) -> list[dict[str, Any]]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        tracks: list[dict[str, Any]] = []
        if not results:
            return tracks

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return tracks

        ids = boxes.id.int().cpu().tolist()
        xyxy_list = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()

        for track_id, xyxy, score, cls_id in zip(
            ids, xyxy_list, confs, classes
        ):
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            width = x2 - x1
            height = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cls_int = int(cls_id)
            tracks.append(
                {
                    "track_id": int(track_id),
                    "xyxy": [x1, y1, x2, y2],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                    "area": width * height,
                    "conf": float(score),
                    "cls": cls_int,
                    "cls_name": self.model.names.get(cls_int, str(cls_int)),
                }
            )

        return tracks
