"""Chuẩn bị dữ liệu: đọc annotation -> records.json, chia split, xuất dataset YOLO.

    python scripts/prepare_data.py --dataset configs/dataset.yaml
"""
import argparse
import logging

from _common import ROOT, setup_logging

from src.data import (
    export_yolo_dataset,
    load_records_from_config,
    make_splits,
    save_records,
    save_splits,
    select_records,
)
from src.utils import load_yaml

log = logging.getLogger("fruitcount")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "configs/dataset.yaml"))
    ap.add_argument("--no-yolo", action="store_true", help="không xuất dataset định dạng YOLO")
    args = ap.parse_args()
    setup_logging()

    cfg = load_yaml(args.dataset)
    records = load_records_from_config(cfg)
    if not records:
        raise SystemExit("Không tìm thấy ảnh có nhãn. Kiểm tra `paths` trong configs/dataset.yaml.")

    sp = cfg.get("split", {})
    splits = make_splits(
        records,
        val=sp.get("val", 0.15),
        test=sp.get("test", 0.15),
        seed=sp.get("seed", 42),
        stratify_by_count=sp.get("stratify_by_count", True),
    )
    out = ROOT / cfg.get("processed_dir", "data/processed")
    save_records(records, out / "records.json")
    save_splits(splits, out / "splits.json")

    for name, ids in splits.items():
        recs = select_records(records, ids)
        counts = [r.count for r in recs]
        log.info(
            "%-5s %4d ảnh | quả/ảnh: min=%d  TB=%.1f  max=%d | tổng=%d",
            name, len(recs), min(counts, default=0), sum(counts) / max(1, len(counts)), max(counts, default=0), sum(counts),
        )

    if not args.no_yolo:
        by_split = {k: select_records(records, v) for k, v in splits.items()}
        yaml_path = export_yolo_dataset(
            by_split, out / "yolo", cfg.get("classes"), symlink=cfg.get("yolo_export", {}).get("symlink", False)
        )
        log.info("Đã xuất dataset YOLO: %s", yaml_path)
    log.info("Đã lưu records/splits vào %s", out)


if __name__ == "__main__":
    main()
