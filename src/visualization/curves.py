"""Biểu đồ: training curves, dự đoán vs thực tế, so sánh các model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path: Optional[str | Path]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Sequence[dict], out_path: Optional[str | Path] = None, title: str = ""):
    """Hai panel: loss huấn luyện và MAE/mAP trên tập val theo epoch."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h["epoch"] for h in history]
    loss_keys = [k for k in history[0] if k.startswith("train_")] if history else []
    for k in loss_keys:
        a1.plot(ep, [h.get(k, np.nan) for h in history], label=k.replace("train_", ""))
    a1.set(xlabel="epoch", ylabel="loss", title="Training loss")
    a1.legend()
    ve = [h["epoch"] for h in history if "val_mae" in h]
    if ve:
        a2.plot(ve, [h["val_mae"] for h in history if "val_mae" in h], "o-", color="tab:red", label="val MAE")
        a2.set(xlabel="epoch", ylabel="MAE (quả/ảnh)", title="Validation")
        if any("val_map50" in h for h in history):
            ax = a2.twinx()
            ax.plot(ve, [h.get("val_map50", np.nan) for h in history if "val_mae" in h], "s--",
                    color="tab:blue", label="val mAP50")
            ax.set_ylabel("mAP@0.5")
            ax.set_ylim(0, 1)
        a2.legend(loc="upper right")
    if title:
        fig.suptitle(title)
    _save(fig, out_path)
    return fig


def plot_pred_vs_gt(
    pred: Sequence[float], gt: Sequence[float], out_path: Optional[str | Path] = None, title: str = ""
):
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(g, p, s=14, alpha=0.7)
    lim = max(g.max() if len(g) else 1, p.max() if len(p) else 1) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Số quả thực tế", ylabel="Số quả dự đoán", title=title)
    _save(fig, out_path)
    return fig


def plot_model_comparison(
    metrics: dict[str, dict[str, float]], keys: Sequence[str] = ("mae", "rmse"),
    out_path: Optional[str | Path] = None,
):
    """Biểu đồ cột so sánh các model theo từng chỉ số (mỗi chỉ số một panel)."""
    names = list(metrics)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8), squeeze=False)
    for ax, k in zip(axes[0], keys):
        vals = [metrics[n].get(k, np.nan) for n in names]
        ax.bar(range(len(names)), vals, color="tab:blue")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_title(k)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_error_by_bin(rows_by_model: dict[str, list[dict]], out_path: Optional[str | Path] = None):
    """MAE theo nhóm số quả thực tế cho từng model (đường cong so sánh khi mật độ tăng)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, rows in rows_by_model.items():
        ax.plot([r["gt_count_bin"] for r in rows], [r["mae"] for r in rows], "o-", label=name)
    ax.set(xlabel="Số quả thực tế / ảnh", ylabel="MAE", title="Sai số theo mật độ quả")
    ax.legend()
    _save(fig, out_path)
    return fig
