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
        val_records=None,
    ) -> Path:
        """Huấn luyện bằng Ultralytics; trả về đường dẫn best.pt."""
        from ultralytics import YOLO

        out_dir = Path(out_dir).resolve()
        last = out_dir / "weights" / "last.pt"
        from ..utils.resources import guard_disk, state_bytes
        def check_checkpoint_space(trainer):
            # Adam/EMA and serializer buffers; conservative for YOLO's FP16 saves.
            guard_disk(state_bytes(trainer.model.state_dict()) * 8 + 8 * 1024**2)
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
            resumed = YOLO(str(last))
            resumed.add_callback('on_train_epoch_start', check_checkpoint_space)
            if val_records is not None:
                self._add_counting_callback(resumed, val_records, out_dir)
            resumed.train(resume=True, data=str(Path(data_yaml).resolve()), device=self.device, epochs=target_epochs)
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
                cache=False,
                save_period=-1,
            )
            self.model.add_callback('on_train_epoch_start', check_checkpoint_space)
            if val_records is not None:
                self._add_counting_callback(self.model, val_records, out_dir)
            self.model.train(**args)
        best = out_dir / "weights" / "best.pt"
        if best.exists():
            self.load(str(best))
        return best

    def _add_counting_callback(self, yolo, records, out_dir):
        """Measure real counting accuracy on validation after every YOLO epoch."""
        import copy
        import csv
        from ..training.validation import validate
        from ..utils.io import read_json, write_json
        from ..visualization.curves import plot_loss_accuracy

        records = list(records)
        if not records:
            raise ValueError('Counting accuracy requires a non-empty validation split')
        folder = Path(out_dir) / 'training'
        history_path = folder / 'history.json'
        history = read_json(history_path) if history_path.exists() else []

        def record_epoch(trainer):
            # A separate copy prevents prediction/fusion from modifying training weights.
            counter = copy.copy(self)
            counter.model = copy.copy(yolo)
            counter.model.overrides = dict(yolo.overrides)
            counter.model.model = copy.deepcopy(trainer.ema.ema if trainer.ema else trainer.model).eval()
            counter.model.predictor = None
            row = {'epoch': trainer.epoch}
            row.update({key.replace('/', '_'): float(value) for key, value in
                        trainer.label_loss_items(trainer.tloss, prefix='train').items()})
            row.update({key.replace('/', '_'): float(value) for key, value in
                        trainer.metrics.items() if key.startswith('val/') and 'loss' in key})
            row.update(validate(counter, records, batch_size=1, with_map=False))
            del counter
            history[:] = [entry for entry in history if entry['epoch'] < trainer.epoch]
            history.append(row)
            write_json(history, history_path)
            keys = list(dict.fromkeys(key for entry in history for key in entry))
            with (folder / 'history.csv').open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(history)
            plot_loss_accuracy(history, folder, self.cfg.get('name', ''))

        yolo.add_callback('on_fit_epoch_end', record_epoch)

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
