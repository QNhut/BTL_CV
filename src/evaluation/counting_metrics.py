"""Chỉ số đếm: so sánh số quả dự đoán với số quả thực tế trên từng ảnh."""
from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def mae(pred, gt) -> float:
    p, g = _arr(pred), _arr(gt)
    return float(np.mean(np.abs(p - g))) if len(g) else float("nan")


def rmse(pred, gt) -> float:
    p, g = _arr(pred), _arr(gt)
    return float(np.sqrt(np.mean((p - g) ** 2))) if len(g) else float("nan")


def mape(pred, gt) -> float:
    """Sai số phần trăm tuyệt đối trung bình (%), chỉ tính ảnh có gt > 0."""
    p, g = _arr(pred), _arr(gt)
    m = g > 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(p[m] - g[m]) / g[m]) * 100.0)


def bias(pred, gt) -> float:
    """Sai số có dấu trung bình (> 0: đếm dư, < 0: đếm thiếu)."""
    p, g = _arr(pred), _arr(gt)
    return float(np.mean(p - g)) if len(g) else float("nan")


def total_relative_error(pred, gt) -> float:
    """(tổng dự đoán - tổng thực) / tổng thực, hữu ích cho ước lượng sản lượng cả vườn."""
    p, g = _arr(pred), _arr(gt)
    return float((p.sum() - g.sum()) / g.sum()) if g.sum() > 0 else float("nan")


def r2_score(pred, gt) -> float:
    p, g = _arr(pred), _arr(gt)
    ss_tot = float(np.sum((g - g.mean()) ** 2)) if len(g) else 0.0
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - np.sum((g - p) ** 2) / ss_tot)


def accuracy_within(pred, gt, tol: float = 0.0) -> float:
    """Tỉ lệ ảnh có |pred - gt| <= tol (làm tròn pred về số nguyên)."""
    p, g = np.rint(_arr(pred)), _arr(gt)
    return float(np.mean(np.abs(p - g) <= tol)) if len(g) else float("nan")


def counting_report(pred, gt) -> dict[str, float]:
    p, g = _arr(pred), _arr(gt)
    return {
        "n_images": int(len(g)),
        "mae": mae(p, g),
        "rmse": rmse(p, g),
        "mape_pct": mape(p, g),
        "bias": bias(p, g),
        "total_rel_error": total_relative_error(p, g),
        "r2": r2_score(p, g),
        "acc_exact": accuracy_within(p, g, 0),
        "acc_within_1": accuracy_within(p, g, 1),
        "acc_within_3": accuracy_within(p, g, 3),
        "gt_total": float(g.sum()),
        "pred_total": float(p.sum()),
    }
