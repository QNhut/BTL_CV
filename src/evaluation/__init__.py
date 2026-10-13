"""Đánh giá: mAP, MAE/RMSE, FPS và phân tích lỗi."""
from .counting_metrics import (
    accuracy_within,
    bias,
    counting_report,
    mae,
    mape,
    r2_score,
    rmse,
    total_relative_error,
)
from .detection_metrics import ap_at_iou, average_precision, evaluate_detection
from .error_analysis import (
    error_by_count_bin,
    per_image_table,
    tune_count_threshold,
    worst_cases,
    write_csv,
)
from .speed import measure_speed

__all__ = [
    "accuracy_within",
    "ap_at_iou",
    "average_precision",
    "bias",
    "counting_report",
    "error_by_count_bin",
    "evaluate_detection",
    "evaluate_model",
    "run_inference",
    "summarize",
    "mae",
    "mape",
    "measure_speed",
    "per_image_table",
    "r2_score",
    "rmse",
    "total_relative_error",
    "tune_count_threshold",
    "worst_cases",
    "write_csv",
]


def __getattr__(name):  # evaluator kéo theo src.models nên import lười
    if name in {"evaluate_model", "run_inference", "summarize"}:
        from . import evaluator

        return getattr(evaluator, name)
    raise AttributeError(name)
