"""Sinh density map ground-truth từ box (tâm box = một điểm quả).

Mỗi quả đóng góp một Gaussian có tổng = 1, nên tổng density map = số quả
(kể cả khi quả sát biên: cửa sổ Gaussian được chuẩn hóa lại sau khi cắt).
"""
from __future__ import annotations

import math

import numpy as np


def boxes_to_points(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def boxes_to_density(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    downsample: int = 8,
    sigma_scale: float = 0.25,
    min_sigma: float = 1.0,
    fixed_sigma: float | None = None,
) -> np.ndarray:
    """Trả về density map (ceil(H/ds), ceil(W/ds)) float32 với tổng = số box nằm trong ảnh.

    sigma (đơn vị: ô density) = sigma_scale * min(w, h)/downsample, tối thiểu `min_sigma`,
    hoặc `fixed_sigma` nếu được đặt.
    """
    h, w = image_shape
    dh, dw = math.ceil(h / downsample), math.ceil(w / downsample)
    dens = np.zeros((dh, dw), np.float32)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return dens

    pts = boxes_to_points(boxes) / downsample
    sizes = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) / downsample
    for (px, py), sz in zip(pts, sizes):
        cx, cy = int(math.floor(px)), int(math.floor(py))
        if not (0 <= cx < dw and 0 <= cy < dh):
            continue
        sigma = fixed_sigma if fixed_sigma else max(min_sigma, sigma_scale * float(sz))
        r = max(1, int(math.ceil(3 * sigma)))
        x0, x1 = max(0, cx - r), min(dw, cx + r + 1)
        y0, y1 = max(0, cy - r), min(dh, cy + r + 1)
        xs = np.arange(x0, x1, dtype=np.float32) + 0.5 - px
        ys = np.arange(y0, y1, dtype=np.float32) + 0.5 - py
        g = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2 * sigma**2))
        total = g.sum()
        if total <= 0:  # sigma quá nhỏ so với offset: đặt vào ô chứa tâm
            dens[cy, cx] += 1.0
            continue
        dens[y0:y1, x0:x1] += g / total
    return dens
