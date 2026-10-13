"""Nạp trọng số và đếm quả cho ảnh / thư mục ảnh; dùng chung cho CLI và demo."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..data.annotations import list_images
from ..models.base import BaseCounter, CountResult
from ..models.registry import build_model, load_model_config
from ..utils.config import deep_merge
from ..utils.io import load_image


class CountingPredictor:
    def __init__(self, model: BaseCounter):
        self.model = model

    @classmethod
    def from_config(
        cls,
        model_cfg_path: str | Path,
        weights: Optional[str | Path] = None,
        device: Optional[str] = None,
        overrides: Optional[dict] = None,
    ) -> "CountingPredictor":
        cfg = load_model_config(model_cfg_path)
        if overrides:
            cfg = deep_merge(cfg, overrides)
        model = build_model(cfg, device=device)
        if weights:
            model.load(str(weights))
        return cls(model)

    # ----- cấu hình lúc chạy -----
    def set_count_threshold(self, thr: float) -> None:
        self.model.set_count_threshold(thr)

    def set_tiling(self, enabled: bool, size: Optional[int] = None, overlap: Optional[float] = None) -> None:
        tile = dict(self.model.inference_cfg.get("tile") or {})
        tile["enabled"] = bool(enabled)
        if size:
            tile["size"] = int(size)
        if overlap is not None:
            tile["overlap"] = float(overlap)
        self.model.inference_cfg["tile"] = tile

    # ----- dự đoán -----
    def predict_array(self, image_rgb: np.ndarray) -> CountResult:
        return self.model.predict_image(image_rgb)

    def predict_path(self, path: str | Path) -> CountResult:
        return self.predict_array(load_image(path))

    def predict_folder(self, folder: str | Path) -> dict[str, CountResult]:
        return {p.name: self.predict_path(p) for p in list_images(folder)}

    @staticmethod
    def result_to_dict(result: CountResult, image_name: str = "") -> dict:
        boxes, scores = result.kept()
        d = {
            "image": image_name,
            "count": int(round(result.count)),
            "count_raw": float(result.count),
        }
        if not result.is_density:
            d["boxes_xyxy"] = np.round(boxes, 1).tolist()
            d["scores"] = np.round(scores, 4).tolist()
        return d
