"""Huấn luyện một model theo config.

    python scripts/train.py --model configs/models/csrnet.yaml
    python scripts/train.py --model configs/models/yolov8.yaml --epochs 50
    python scripts/train.py --model configs/models/faster_rcnn.yaml --resume
"""
import argparse
import logging

from _common import ROOT, run_dir, setup_logging

from src.data import export_yolo_dataset, load_split_records
from src.models import build_model, load_model_config
from src.utils import deep_merge, load_yaml, save_yaml, seed_everything
from src.visualization import plot_training_curves

log = logging.getLogger("fruitcount")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="configs/models/<model>.yaml")
    ap.add_argument("--dataset", default=str(ROOT / "configs/dataset.yaml"))
    ap.add_argument("--aug", default=str(ROOT / "configs/augmentation.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs"), help="thư mục gốc chứa kết quả")
    ap.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch trong config")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true", help="tiếp tục từ weights/last.pt")
    args = ap.parse_args()
    setup_logging()

    cfg = load_model_config(args.model)
    ds_cfg = load_yaml(args.dataset)
    aug_cfg = load_yaml(args.aug)
    processed = ROOT / ds_cfg.get("processed_dir", "data/processed")
    out_dir = run_dir(args.out, cfg)

    over: dict = {}
    if cfg["type"] == "yolo":
        if args.epochs:
            over.setdefault("train_args", {})["epochs"] = args.epochs
        if args.batch_size:
            over.setdefault("train_args", {})["batch"] = args.batch_size
    else:
        if args.epochs:
            over.setdefault("train", {})["epochs"] = args.epochs
        if args.batch_size:
            over.setdefault("train", {})["batch_size"] = args.batch_size
    cfg = deep_merge(cfg, over)
    save_yaml(cfg, out_dir / "config.yaml")  # lưu lại cấu hình đã dùng để tái lập kết quả

    training_cfg = (cfg.get("train_args") if cfg["type"] == "yolo" else cfg.get("train")) or {}
    seed_everything(int(training_cfg.get("seed", 42)))
    model = build_model(cfg, device=args.device)

    if cfg["type"] == "yolo":
        data_yaml = processed / "yolo" / "data.yaml"
        if not data_yaml.exists():
            log.info("Chưa có dataset YOLO, đang xuất từ records...")
            by_split = {s: load_split_records(processed, s) for s in ("train", "val", "test")}
            data_yaml = export_yolo_dataset(by_split, processed / "yolo", ds_cfg.get("classes"))
        best = model.fit(data_yaml, out_dir, resume=args.resume)
        log.info("Xong. Trọng số tốt nhất: %s", best)
        return

    from src.training import Trainer  # import trễ vì cần torch

    train_recs = load_split_records(processed, "train")
    val_recs = load_split_records(processed, "val")
    log.info("Train %d ảnh | Val %d ảnh | model %s (%.1fM tham số)",
             len(train_recs), len(val_recs), cfg["name"], model.num_parameters() / 1e6)
    summary = Trainer(model, train_recs, val_recs, aug_cfg, out_dir).fit(resume=args.resume)
    plot_training_curves(summary["history"], out_dir / "training_curves.png", title=cfg["name"])
    log.info("Xong. Val MAE tốt nhất %.3f tại epoch %d -> %s",
             summary["best_val_mae"], summary["best_epoch"], out_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
