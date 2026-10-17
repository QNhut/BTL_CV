"""Adapter YOLOv8 / YOLO11 (Ultralytics). Huấn luyện do Ultralytics đảm nhiệm,
còn suy luận trả về `CountResult` giống các model khác để đánh giá công bằng."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseCounter, CountResult


class YOLOCounter(BaseCounter):
    family = "detection"
    trainer = "ultralytics"

    def __init__(self, cfg: dict, device: Optional[str] = None):
        super().__init__(cfg)
        from ultralytics import YOLO  # import trễ: chỉ cần khi dùng YOLO

        m = cfg.get("model", {}) or {}
        self.device = device or m.get("device")
        self.imgsz = int(m.get("imgsz", 1024))
        self.weights = m.get("weights", "yolov8s.pt")
        self.model = YOLO(self.weights)

    # ----- huấn luyện -----
    def fit(
        self,
        data_yaml: str | Path,
        out_dir: str | Path,
        resume: bool = False,
        overrides: Optional[dict] = None,
    ) -> Path:
        """Huấn luyện bằng Ultralytics; trả về đường dẫn best.pt."""
        from ultralytics import YOLO

        out_dir = Path(out_dir).resolve()
        last = out_dir / "weights" / "last.pt"
        if resume:
            if not last.exists():
                raise FileNotFoundError(f"Không có checkpoint để resume: {last}")
            import torch

            checkpoint = torch.load(last, map_location="cpu", weights_only=False)
            target_epochs = int((overrides or {}).get("epochs", (self.cfg.get("train_args") or {}).get("epochs", 100)))
            epoch = int(checkpoint.get("epoch", -1))
            best = out_dir / "weights" / "best.pt"
            if (epoch == -1 or epoch + 1 >= target_epochs) and best.exists():
                self.load(str(best))
                return best
            if epoch == -1:
                raise ValueError("This YOLO checkpoint has no optimizer state; use a training checkpoint to resume.")
            del checkpoint
            YOLO(str(last)).train(resume=True, data=str(Path(data_yaml).resolve()), device=self.device, epochs=target_epochs)
        else:
            args = dict(self.cfg.get("train_args", {}) or {})
            args.update(overrides or {})
            args.update(
                data=str(data_yaml),
                imgsz=self.imgsz,
                project=str(out_dir.parent),
                name=out_dir.name,
                exist_ok=True,
                single_cls=True,
                device=self.device,
                save=True,
            )
            self.model.train(**args)
        best = out_dir / "weights" / "best.pt"
        if best.exists():
            self.load(str(best))
        return best

    # ----- suy luận -----
    def _predict_raw(self, images: list[np.ndarray]) -> list[CountResult]:
        if not images:
            return []
        bgr = [np.ascontiguousarray(im[..., ::-1]) for im in images]  # Ultralytics coi mảng numpy là BGR
        results = self.model.predict(
            bgr,
            imgsz=self.imgsz,
            conf=max(self.min_score, 1e-3),
            iou=float(self.inference_cfg.get("nms_iou", 0.5)),
            max_det=int(self.inference_cfg.get("max_det", 1000)),
            device=self.device,
            verbose=False,
        )
        out = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = r.boxes.conf.cpu().numpy().astype(np.float32)
            out.append(CountResult(boxes=boxes, scores=scores, score_thr=self.count_threshold))
        return out

    def load(self, weights: str) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.weights = str(weights)

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.model.parameters()))
