"""Đăng ký model theo `type` trong configs/models/*.yaml. Import lười để chỉ cần cài
thư viện của model đang dùng (ví dụ không cần ultralytics khi chạy CSRNet)."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

from ..utils.config import load_yaml
from .base import BaseCounter

_REGISTRY: dict[str, tuple[str, str]] = {
    "yolo": ("yolo", "YOLOCounter"),
    "faster_rcnn": ("faster_rcnn", "FasterRCNNCounter"),
    "mask_rcnn": ("mask_rcnn", "MaskRCNNCounter"),
    "csrnet": ("csrnet", "CSRNetCounter"),
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def get_model_class(model_type: str) -> type[BaseCounter]:
    if model_type not in _REGISTRY:
        raise KeyError(f"Model type '{model_type}' không có; hiện có: {available_models()}")
    module, cls = _REGISTRY[model_type]
    return getattr(importlib.import_module(f".{module}", __package__), cls)


def build_model(cfg: dict, device: Optional[str] = None) -> BaseCounter:
    """Dựng model từ dict config (nội dung một file configs/models/*.yaml)."""
    return get_model_class(cfg["type"])(cfg, device=device)


def load_model_config(path: str | Path) -> dict:
    cfg = load_yaml(path)
    for key in ("name", "type", "family"):
        if key not in cfg:
            raise KeyError(f"Config model {path} thiếu khoá '{key}'")
    return cfg
