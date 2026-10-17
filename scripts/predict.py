"""Đếm quả cho một ảnh hoặc cả thư mục ảnh, lưu ảnh minh họa và JSON.

    python scripts/predict.py --model configs/models/yolov8.yaml --source anh_cay.jpg
    python scripts/predict.py --model configs/models/csrnet.yaml --source thu_muc_anh/ --out outputs/csrnet
"""
import argparse
import logging
from pathlib import Path

from _common import ROOT, default_weights, run_dir, setup_logging

from src.data.annotations import IMAGE_EXTS, list_images
from src.inference import CountingPredictor
from src.utils import load_image, load_yaml, save_image, write_json
from src.visualization import render_result

log = logging.getLogger("fruitcount")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--source", required=True, help="ảnh hoặc thư mục ảnh")
    ap.add_argument("--out", default=None, help="default: <runs>/<model>/predictions")
    ap.add_argument("--runs", default=str(ROOT / "runs"), help="nơi tìm best.pt mặc định")
    ap.add_argument("--count-threshold", type=float, default=None)
    ap.add_argument("--tile", action="store_true")
    ap.add_argument("--tile-size", type=int, default=None)
    ap.add_argument("--tile-overlap", type=float, default=None)
    ap.add_argument("--show-scores", action="store_true")
    ap.add_argument("--no-vis", action="store_true", help="chỉ ghi JSON, không lưu ảnh minh họa")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    setup_logging()

    pred = CountingPredictor.from_config(args.model, device=args.device)
    weights = args.weights or default_weights(args.runs, pred.model.cfg)
    pred.model.load(str(weights))
    if args.tile:
        pred.set_tiling(True, args.tile_size, args.tile_overlap)
    if args.count_threshold is not None:
        pred.set_count_threshold(args.count_threshold)

    src = Path(args.source)
    paths = [src] if src.is_file() else list_images(src)
    if src.is_file() and src.suffix.lower() not in IMAGE_EXTS:
        raise SystemExit(f"Không phải file ảnh: {src}")
    out = Path(args.out) if args.out else run_dir(args.runs, pred.model.cfg) / "predictions"
    rows = []
    for p in paths:
        img = load_image(p)
        res = pred.predict_array(img)
        rows.append(pred.result_to_dict(res, p.name))
        log.info("%-30s -> %d quả", p.name, rows[-1]["count"])
        if not args.no_vis:
            save_image(out / f"{p.stem}_count.jpg", render_result(img, res, args.show_scores))
    write_json({"model": pred.model.cfg["name"], "results": rows}, out / "predictions.json")
    log.info("Tổng: %d ảnh, %d quả. Kết quả: %s", len(rows), sum(r["count"] for r in rows), out)


if __name__ == "__main__":
    main()
