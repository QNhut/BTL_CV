"""Suy luận và hậu xử lý. `CountingPredictor` được import lười để tránh vòng import với `src.models`."""
from .postprocess import (
    box_iou_matrix,
    filter_by_score,
    merge_tile_detections,
    nms,
    tile_slices,
)

__all__ = [
    "box_iou_matrix",
    "filter_by_score",
    "merge_tile_detections",
    "nms",
    "tile_slices",
    "CountingPredictor",
]


def __getattr__(name):
    if name == "CountingPredictor":
        from .predictor import CountingPredictor

        return CountingPredictor
    raise AttributeError(name)
