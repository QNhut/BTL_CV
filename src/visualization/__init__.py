"""Trực quan hóa: training curves, bbox / mask / density map, so sánh model."""
from .curves import plot_error_by_bin, plot_model_comparison, plot_pred_vs_gt, plot_training_curves
from .overlays import (
    draw_boxes,
    draw_count_banner,
    overlay_density,
    overlay_masks,
    render_result,
)

__all__ = [
    "draw_boxes",
    "draw_count_banner",
    "overlay_density",
    "overlay_masks",
    "render_result",
    "plot_error_by_bin",
    "plot_model_comparison",
    "plot_pred_vs_gt",
    "plot_training_curves",
]
