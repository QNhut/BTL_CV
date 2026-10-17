"""Giao diện chung cho mọi model đếm quả.

Mọi adapter (YOLO, Faster R-CNN, Mask R-CNN, CSRNet) trả về `CountResult`, nên
phần đánh giá / demo không cần biết model bên dưới là loại nào.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..inference.postprocess import merge_tile_detections, tile_slices


@dataclass
class CountResult:
    """Kết quả cho một ảnh.

    - Model phát hiện / segmentation: `boxes`, `scores` (mọi box >= min_score),
      số đếm = số box có score >= `score_thr`.
    - Model density: `density` (map) và `density_count` (= tổng density).
    """

    boxes: Optional[np.ndarray] = None       # (N, 4) xyxy
    scores: Optional[np.ndarray] = None      # (N,)
    masks: Optional[np.ndarray] = None       # (M, H, W) bool, ứng với box có score >= score_thr
    density: Optional[np.ndarray] = None     # (h, w)
    density_count: Optional[float] = None
    score_thr: float = 0.5

    @property
    def is_density(self) -> bool:
        return self.density_count is not None

    def count_at(self, thr: Optional[float] = None) -> float:
        if self.is_density:
            return float(self.density_count)
        if self.scores is None:
            return 0.0
        t = self.score_thr if thr is None else thr
        return float((self.scores >= t).sum())

    @property
    def count(self) -> float:
        return self.count_at()

    def kept(self, thr: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """Box và điểm sau khi áp ngưỡng đếm."""
        if self.boxes is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        t = self.score_thr if thr is None else thr
        m = self.scores >= t
        return self.boxes[m], self.scores[m]


class BaseCounter(ABC):
    name: str = "base"
    family: str = "detection"        # detection | segmentation | density
    supports_tiling: bool = True
    supports_box_metrics: bool = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.inference_cfg: dict = dict(cfg.get("inference", {}) or {})

    # ----- ngưỡng -----
    @property
    def min_score(self) -> float:
        return float(self.inference_cfg.get("min_score") or 0.0)

    @property
    def count_threshold(self) -> float:
        thr = self.inference_cfg.get("count_threshold")
        return 0.5 if thr is None else float(thr)

    def set_count_threshold(self, thr: float) -> None:
        self.inference_cfg["count_threshold"] = float(thr)

    # ----- dự đoán -----
    @abstractmethod
    def _predict_raw(self, images: list[np.ndarray]) -> list[CountResult]:
        """Dự đoán trên ảnh RGB uint8 (H,W,3), không tile."""

    def predict_images(self, images: Sequence[np.ndarray]) -> list[CountResult]:
        tile = self.inference_cfg.get("tile") or {}
        images = list(images)
        if tile.get("enabled") and self.supports_tiling and self.family != "density":
            return [self._predict_tiled(im, tile) for im in images]
        return self._predict_raw(images)

    def predict_image(self, image: np.ndarray) -> CountResult:
        return self.predict_images([image])[0]

    def _predict_tiled(self, image: np.ndarray, tile_cfg: dict) -> CountResult:
        h, w = image.shape[:2]
        size = int(tile_cfg.get("size", 640))
        slices = tile_slices(h, w, size, float(tile_cfg.get("overlap", 0.2)))
        tiles = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in slices]
        outs = self._predict_raw(tiles)
        per_tile = [(o.boxes, o.scores) for o in outs]
        boxes, scores = merge_tile_detections(
            per_tile, [(s[0], s[1]) for s in slices], float(self.inference_cfg.get("nms_iou", 0.5))
        )
        return CountResult(boxes=boxes, scores=scores, score_thr=self.count_threshold)

    # ----- vào/ra trọng số -----
    @abstractmethod
    def load(self, weights: str) -> None:
        """Nạp trọng số (file .pt/.pth)."""

    def num_parameters(self) -> Optional[int]:
        return None
