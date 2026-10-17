"""Gộp kết quả các model thành bảng so sánh + biểu đồ (dùng cho báo cáo).

    python scripts/compare.py --split test
"""
import argparse
import logging
import math
from pathlib import Path

from _common import ROOT, setup_logging

from src.evaluation import write_csv
from src.utils import read_json
from src.visualization import plot_error_by_bin, plot_model_comparison
from src.utils.run_layout import evaluation_reports

log = logging.getLogger("fruitcount")

COUNT_COLUMNS = [
    "mae", "rmse", "mape_pct", "bias", "total_rel_error", "r2",
    "acc_exact", "acc_within_1", "acc_within_3",
]
DETECTION_COLUMNS = [
    "map50", "map75", "map", "precision_at_best_f1", "recall_at_best_f1",
    "best_f1", "best_f1_score_thr",
]
SPEED_COLUMNS = ["fps", "latency_ms_mean", "latency_ms_median", "latency_ms_p95",
                 "latency_ms_std", "latency_ms_min", "latency_ms_max", "params_million",
                 "model_tensor_mb", "gpu_peak_allocated_mb", "gpu_inference_extra_mb",
                 "benchmark_device", "benchmark_hardware", "benchmark_batch_size",
                 "benchmark_runs", "benchmark_warmup", "cpu_threads", "torch_version", "parameter_dtype"]
RUNTIME_COLUMNS = ["model_load_s", "dataset_inference_s", "dataset_images_per_s", "dataset_batch_size"]
COLUMNS = ["model", "family", "protocol", "n_images", *COUNT_COLUMNS, *DETECTION_COLUMNS,
           *SPEED_COLUMNS, "checkpoint_mb", *RUNTIME_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="mặc định: <runs>/comparison")
    args = ap.parse_args()
    setup_logging()

    runs = Path(args.runs)
    out = Path(args.out) if args.out else runs / "comparison"
    rows, metrics, bins = [], {}, {}
    for f in evaluation_reports(runs, args.split):
        r = read_json(f)
        tag = f.stem[len(f"metrics_{args.split}"):]
        name = r["model"] + (f" ({tag.lstrip('_')})" if tag else " (tile)" if r.get("tiling") else "")
        c, d, s = r["counting"], r.get("detection", {}), r.get("speed", {})
        row = {
            "model": name, "family": r["family"],
            'protocol': r.get('protocol', 'unspecified'),
            "n_images": c.get("n_images", r.get("n_images", "")),
            **{k: c.get(k, "") for k in COUNT_COLUMNS},
            **{k: d.get(k, "") for k in DETECTION_COLUMNS},
            **{k: s.get(k, "") for k in SPEED_COLUMNS},
            'checkpoint_mb': r.get('storage', {}).get('checkpoint_mb', ''),
            **{k: r.get('runtime', {}).get(k, '') for k in RUNTIME_COLUMNS},
        }
        rows.append(row)
        metrics[name] = {k: v for k, v in row.items() if isinstance(v, float)}
        bins[name] = r.get("error_by_count_bin", [])
    if not rows:
        raise SystemExit(f"Chưa có metrics_{args.split}*.json trong {runs}; hãy chạy scripts/evaluate.py trước.")

    rows.sort(key=lambda r: r["mae"] if isinstance(r["mae"], (int, float)) and math.isfinite(r["mae"]) else math.inf)
    write_csv([{k: r[k] for k in COLUMNS} for r in rows], out / f"comparison_{args.split}.csv")
    deployment_columns = ['model', 'protocol', 'mae', 'acc_within_1', 'fps', 'latency_ms_mean',
                          'latency_ms_p95', 'checkpoint_mb', 'model_tensor_mb',
                          'gpu_peak_allocated_mb', 'gpu_inference_extra_mb', 'params_million',
                          *RUNTIME_COLUMNS, 'benchmark_device', 'benchmark_hardware',
                          'benchmark_runs', 'benchmark_warmup', 'parameter_dtype']
    write_csv([{key: row[key] for key in deployment_columns} for row in rows],
              out / f'deployment_{args.split}.csv')
    plot_model_comparison(metrics, ("mae", "rmse", "mape_pct"), out / f"comparison_{args.split}.png")
    plot_error_by_bin(bins, out / f"error_by_count_{args.split}.png")
    if any('fps' in values for values in metrics.values()):
        plot_model_comparison(metrics, ('fps', 'latency_ms_p95', 'checkpoint_mb'),
                              out / f'deployment_{args.split}.png')

    def fmt(v):
        if v == "" or v is None or isinstance(v, (int, float)) and not math.isfinite(v):
            return "—"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in COLUMNS) + " |" for r in rows]
    lines += [
        "", "Accuracy, precision, recall, F1 and mAP use a 0–1 scale; mape_pct uses percent.",
        "total_rel_error is a signed ratio. Latency is in milliseconds; params_million is in millions.",
        "Precision/recall/F1 are measured at best F1 with IoU 0.5, not necessarily at the counting threshold.",
        "— means unavailable, undefined, or not applicable (density models have no detection metrics).",
        "FPS measures sequential batch-1 inference on loaded images, including preprocessing and postprocessing; excludes disk I/O and model loading.",
        "dataset_inference_s includes image reads and inference on the evaluation split; excludes threshold tuning and metric calculations.",
        "All *_mb columns use MiB (1024^2 bytes). GPU allocation covers this process, not total device use; extra memory is peak minus post-warmup baseline.",
        "Compare on the same device, image set, precision, software and tiling settings. Model loading may include pretrained weight downloads.",
    ]
    (out / f"comparison_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    deployment_lines = ['| ' + ' | '.join(deployment_columns) + ' |',
                        '|' + '---|' * len(deployment_columns)]
    deployment_lines += ['| ' + ' | '.join(fmt(row[key]) for key in deployment_columns) + ' |' for row in rows]
    deployment_lines += lines[-8:]
    (out / f'deployment_{args.split}.md').write_text('\n'.join(deployment_lines) + '\n', encoding='utf-8')
    log.info("\n" + "\n".join(lines))
    log.info("Đã ghi bảng và biểu đồ vào %s", out)


if __name__ == "__main__":
    main()
