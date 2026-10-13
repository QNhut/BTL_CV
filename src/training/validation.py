"""Validation trong lúc huấn luyện: chạy model trên tập val, trả về MAE/RMSE (+ mAP@0.5)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.annotations import ImageRecord
from ..evaluation.counting_metrics import mae, rmse
from ..evaluation.detection_metrics import evaluate_detection
from ..evaluation.error_analysis import tune_count_threshold
from ..evaluation.evaluator import run_inference
from ..models.base import BaseCounter


def validate(
    model: BaseCounter, records: Sequence[ImageRecord], batch_size: int = 2, with_map: bool = True
) -> dict[str, float]:
    """Với model phát hiện/segmentation: MAE được lấy tại ngưỡng đếm tốt nhất trên val
    (xem `val_threshold`) để so sánh các epoch công bằng, không phụ thuộc ngưỡng cố định."""
    records = list(records)
    results = run_inference(model, records, batch_size)
    gt = np.array([r.count for r in records], np.float64)
    metrics: dict[str, float] = {}
    if model.family == "density":
        pred = np.array([r.count_at() for r in results])
        metrics["val_mae"], metrics["val_rmse"] = mae(pred, gt), rmse(pred, gt)
    else:
        thr, best_mae = tune_count_threshold(results, gt)
        pred = np.array([r.count_at(thr) for r in results])
        metrics.update(val_mae=best_mae, val_rmse=rmse(pred, gt), val_threshold=thr)
        if with_map:
            metrics["val_map50"] = evaluate_detection(
                [r.boxes for r in results], [r.scores for r in results], [r.boxes for r in records], iou_thrs=(0.5,)
            )["map50"]
    return metrics
