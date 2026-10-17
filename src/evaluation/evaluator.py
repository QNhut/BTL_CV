"""Chạy model trên một tập records và tổng hợp mAP / MAE / RMSE / FPS / phân tích lỗi."""
from __future__ import annotations

from typing import Optional, Sequence
import time

import numpy as np

from ..data.annotations import ImageRecord
from ..models.base import BaseCounter, CountResult
from ..utils.io import load_image
from .counting_metrics import counting_report
from .detection_metrics import evaluate_detection
from .error_analysis import error_by_count_bin, per_image_table, tune_count_threshold
from .speed import measure_speed, _sync


def run_inference(
    model: BaseCounter, records: Sequence[ImageRecord], batch_size: int = 4, progress: bool = False,
    keep_visuals: bool = False,
) -> list[CountResult]:
    results: list[CountResult] = []
    rng = range(0, len(records), batch_size)
    if progress:
        try:
            from tqdm import tqdm

            rng = tqdm(rng, desc="inference", leave=False)
        except ImportError:
            pass
    for i in rng:
        imgs = [load_image(r.image_path) for r in records[i : i + batch_size]]
        predictions = model.predict_images(imgs)
        if not keep_visuals:
            for result in predictions:
                result.masks = None  # Counting/box metrics do not need full-resolution masks.
                result.density = None  # Density count is already stored separately.
        results.extend(predictions)
    return results


def evaluate_model(
    model: BaseCounter,
    records: Sequence[ImageRecord],
    *,
    batch_size: int = 4,
    tune_on: Optional[Sequence[ImageRecord]] = None,
    with_map: bool = True,
    with_speed: bool = False,
    speed_images: int = 10,
    speed_runs: int = 50,
    speed_warmup: int = 5,
    progress: bool = False,
    return_results: bool = False,
) -> dict:
    """Đánh giá đầy đủ một model.

    tune_on: nếu có (thường là tập val) và model là loại phát hiện, chọn ngưỡng đếm tối ưu
             trên tập đó rồi áp vào đánh giá — tránh tune ngưỡng trực tiếp trên test.
    """
    records = list(records)
    out: dict = {"model": model.cfg.get("name"), "family": model.family, "n_images": len(records)}
    out['protocol'] = model.cfg.get('protocol', 'finetuned')
    out['prompt'] = model.cfg.get('model', {}).get('prompt')

    if tune_on is not None and model.family != "density":
        tune_records = list(tune_on)
        tune_res = run_inference(model, tune_records, batch_size, progress)
        thr, tuned_mae = tune_count_threshold(tune_res, [r.count for r in tune_records])
        model.set_count_threshold(thr)
        out["threshold_tuning"] = {"best_threshold": thr, "val_mae": tuned_mae, "n_images": len(tune_records)}

    _sync()
    started = time.perf_counter()
    results = run_inference(model, records, batch_size, progress, keep_visuals=return_results)
    _sync()
    elapsed = time.perf_counter() - started
    out['runtime'] = {'dataset_inference_s': elapsed,
                      'dataset_images_per_s': len(records) / elapsed if elapsed > 0 else None,
                      'dataset_batch_size': batch_size}
    gt = np.array([r.count for r in records], np.float64)
    pred = np.array([res.count_at() for res in results], np.float64)

    if model.family != "density":
        out["count_threshold"] = model.count_threshold
    out["counting"] = counting_report(pred, gt)
    out["error_by_count_bin"] = error_by_count_bin(pred, gt)
    out["per_image"] = per_image_table([r.image_id for r in records], pred, gt)

    if with_map and model.family != "density" and model.supports_box_metrics:
        out["detection"] = evaluate_detection(
            [res.boxes for res in results], [res.scores for res in results], [r.boxes for r in records]
        )

    if with_speed and records:
        imgs = [load_image(r.image_path) for r in records[:speed_images]]
        out["speed"] = measure_speed(model, imgs, runs=speed_runs, warmup=speed_warmup)

    return (out, results) if return_results else out


def summarize(report: dict) -> str:
    """Một dòng tóm tắt dễ đọc cho log."""
    c = report["counting"]
    s = f"{report['model']}: MAE={c['mae']:.2f} RMSE={c['rmse']:.2f} MAPE={c['mape_pct']:.1f}% R2={c['r2']:.3f}"
    if "detection" in report:
        d = report["detection"]
        s += f" mAP50={d['map50']:.3f} mAP50-95={d['map']:.3f}"
    if "speed" in report:
        s += f" FPS={report['speed']['fps']:.1f}"
    return s
