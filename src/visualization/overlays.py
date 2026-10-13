"""Vẽ box, mask và density map lên ảnh; tất cả hàm nhận/trả ảnh RGB uint8."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..models.base import CountResult

GREEN = (46, 204, 113)
RED = (231, 76, 60)


def draw_count_banner(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    scale = max(0.6, min(out.shape[:2]) / 700)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(10 * scale)
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    color=GREEN,
    thickness: Optional[int] = None,
    show_scores: bool = False,
) -> np.ndarray:
    out = image.copy()
    t = thickness or max(1, int(round(min(out.shape[:2]) / 400)))
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, t)
        if show_scores and scores is not None:
            cv2.putText(out, f"{scores[i]:.2f}", (int(x1), max(0, int(y1) - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def overlay_masks(image: np.ndarray, masks: np.ndarray, alpha: float = 0.45, seed: int = 0) -> np.ndarray:
    """Tô màu ngẫu nhiên cho từng mask instance (masks: [M,H,W] bool)."""
    out = image.astype(np.float32).copy()
    rng = np.random.default_rng(seed)
    for m in np.asarray(masks, bool):
        color = rng.integers(60, 255, size=3).astype(np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def overlay_density(image: np.ndarray, density: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Phủ heatmap density lên ảnh (density tự resize về kích thước ảnh)."""
    h, w = image.shape[:2]
    d = cv2.resize(np.asarray(density, np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    d = d / d.max() if d.max() > 0 else d
    heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    weight = (alpha * np.clip(d * 2, 0, 1))[..., None]  # vùng density thấp trong suốt
    return (image * (1 - weight) + heat * weight).clip(0, 255).astype(np.uint8)


def render_result(image: np.ndarray, result: CountResult, show_scores: bool = False) -> np.ndarray:
    """Hình minh họa chuẩn: box/mask hoặc heatmap + dòng số quả đếm được."""
    if result.is_density:
        out = overlay_density(image, result.density)
    else:
        out = image
        if result.masks is not None and len(result.masks):
            out = overlay_masks(out, result.masks)
        boxes, scores = result.kept()
        out = draw_boxes(out, boxes, scores, show_scores=show_scores)
    return draw_count_banner(out, f"Fruits: {int(round(result.count))}")
