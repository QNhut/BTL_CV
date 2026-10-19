"""Phân tích lỗi: sai số theo mật độ quả, ảnh tệ nhất, và chọn ngưỡng đếm tối ưu."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .counting_metrics import mae, rmse

DEFAULT_BINS = (0, 10, 25, 50, 100, 200, 10**9)


def tune_count_threshold(
    results: Sequence, gt_counts: Sequence[float], grid: Optional[Sequence[float]] = None
) -> tuple[float, float]:
    """Chọn ngưỡng score cho MAE đếm thấp nhất (nên chạy trên tập val, rồi áp cho test).

    `results`: danh sách CountResult của model phát hiện. Trả về (ngưỡng tốt nhất, MAE tại ngưỡng đó).
    """
    grid = list(grid) if grid is not None else [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    gt = np.asarray(gt_counts, np.float64)
    best_thr, best_mae = grid[0], float("inf")
    for thr in grid:
        pred = np.array([r.count_at(thr) for r in results])
        m = mae(pred, gt)
        if m < best_mae:  # hòa thì giữ ngưỡng nhỏ hơn
            best_thr, best_mae = thr, m
    return float(best_thr), float(best_mae)


def per_image_table(image_ids: Sequence[str], pred: Sequence[float], gt: Sequence[float]) -> list[dict]:
    rows = []
    for i, p, g in zip(image_ids, pred, gt):
        rows.append(
            {
                "image_id": i,
                "gt": float(g),
                "pred": float(p),
                "error": float(p - g),
                "abs_error": float(abs(p - g)),
                "rel_error_pct": float(abs(p - g) / g * 100) if g > 0 else float("nan"),
            }
        )
    return rows


def error_by_count_bin(pred: Sequence[float], gt: Sequence[float], bins: Sequence[int] = DEFAULT_BINS) -> list[dict]:
    """MAE/RMSE/bias theo nhóm số quả thực tế: trả lời câu hỏi "quả chồng lấn nhiều thì sao?"."""
    p, g = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (g >= lo) & (g < hi)
        if not m.any():
            continue
        label = f"{lo}-{hi - 1}" if hi < 10**8 else f">={lo}"
        rows.append(
            {
                "gt_count_bin": label,
                "n_images": int(m.sum()),
                "mae": mae(p[m], g[m]),
                "rmse": rmse(p[m], g[m]),
                "bias": float(np.mean(p[m] - g[m])),
            }
        )
    return rows


def worst_cases(rows: list[dict], k: int = 10) -> list[dict]:
    return sorted(rows, key=lambda r: r["abs_error"], reverse=True)[:k]


def write_csv(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows({key: f"{value:.3f}" if isinstance(value, (float, np.floating)) else value
                     for key, value in row.items()} for row in rows)
