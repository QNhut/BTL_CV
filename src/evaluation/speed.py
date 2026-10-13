"""Đo tốc độ suy luận (latency / FPS) và kích thước model."""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def measure_speed(model, images: Sequence[np.ndarray], warmup: int = 2, runs: int = 10) -> dict[str, float]:
    """Đo thời gian predict_images trên từng ảnh (không tính thời gian đọc ảnh)."""
    images = list(images)
    if not images:
        return {}
    for i in range(min(warmup, len(images))):
        model.predict_images([images[i]])
    times = []
    for k in range(runs):
        im = images[k % len(images)]
        _sync()
        t0 = time.perf_counter()
        model.predict_images([im])
        _sync()
        times.append(time.perf_counter() - t0)
    mean = float(np.mean(times))
    out = {
        "latency_ms_mean": mean * 1000,
        "latency_ms_p95": float(np.percentile(times, 95)) * 1000,
        "fps": 1.0 / mean if mean > 0 else float("inf"),
    }
    n = model.num_parameters()
    if n is not None:
        out["params_million"] = n / 1e6
    return out
