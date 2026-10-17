"""Đánh giá một model đã huấn luyện: mAP, MAE, RMSE, FPS, sai số theo mật độ quả.

    python scripts/evaluate.py --model configs/models/yolov8.yaml --split test --speed
    python scripts/evaluate.py --model configs/models/yolov8.yaml --tile   # bật SAHI-style tiling

Với model phát hiện, ngưỡng đếm được chọn trên tập val (tránh tune trên test).
"""
import argparse
import logging

from _common import ROOT, default_weights, run_dir, setup_logging

from src.data import load_split_records
from src.evaluation import evaluate_model, summarize, worst_cases, write_csv
from src.inference import CountingPredictor
from src.utils import load_yaml, write_json
from src.visualization import plot_pred_vs_gt

log = logging.getLogger("fruitcount")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", default=None, help="mặc định: runs/<tên model>/weights/best.pt")
    ap.add_argument("--dataset", default=str(ROOT / "configs/dataset.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tune-on", default="val", help="split dùng chọn ngưỡng đếm ('none' để tắt)")
    ap.add_argument("--count-threshold", type=float, default=None, help="đặt ngưỡng đếm cố định")
    ap.add_argument("--tile", action="store_true", help="bật tiling khi suy luận (model phát hiện)")
    ap.add_argument("--tile-size", type=int, default=None)
    ap.add_argument("--tile-overlap", type=float, default=None)
    ap.add_argument("--speed", action="store_true", help="đo FPS / latency")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="", help="hậu tố cho tên file kết quả, ví dụ _tile")
    args = ap.parse_args()
    setup_logging()

    pred = CountingPredictor.from_config(args.model, weights=None, device=args.device)
    cfg = pred.model.cfg
    weights = args.weights or default_weights(args.out, cfg)
    pred.model.load(str(weights))
    if args.tile:
        pred.set_tiling(True, args.tile_size, args.tile_overlap)
    if args.count_threshold is not None:
        pred.set_count_threshold(args.count_threshold)

    processed = ROOT / load_yaml(args.dataset).get("processed_dir", "data/processed")
    records = load_split_records(processed, args.split)
    tune = None
    if args.tune_on.lower() != "none" and args.count_threshold is None and args.split != args.tune_on:
        tune = load_split_records(processed, args.tune_on)

    report = evaluate_model(
        pred.model, records, batch_size=args.batch_size, tune_on=tune, with_speed=args.speed, progress=True
    )
    log.info(summarize(report))

    out = run_dir(args.out, cfg) / "evaluation"
    suffix = f"{args.split}{args.tag}"
    rows = report.pop("per_image")
    report["worst_cases"] = worst_cases(rows, 10)
    report["weights"] = str(weights)
    report["tiling"] = bool(args.tile)
    write_json(report, out / f"metrics_{suffix}.json")
    write_csv(rows, out / f"per_image_{suffix}.csv")
    write_csv(report["error_by_count_bin"], out / f"error_by_count_{suffix}.csv")
    plot_pred_vs_gt([r["pred"] for r in rows], [r["gt"] for r in rows],
                    out / f"pred_vs_gt_{suffix}.png", title=cfg["name"])
    log.info("Đã ghi kết quả vào %s", out)


if __name__ == "__main__":
    main()
