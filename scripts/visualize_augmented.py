"""Trực quan hóa và so sánh dữ liệu trước & sau tăng cường cho từng chiến lược.

Xuất ra các ảnh so sánh trực quan đa góc nhìn (Side-by-Side):
1. Bounding Boxes (Object Detection - YOLO / Faster R-CNN)
2. Instance Masks (Instance Segmentation - Mask R-CNN)
3. Density Heatmaps (Density Regression - CSRNet)

Cách dùng:
    python scripts/visualize_augmented.py
    python scripts/visualize_augmented.py --num-samples 5 --out runs/augmentation_showcase
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT, setup_logging
from src.data.annotations import masks_to_boxes, read_instance_mask
from src.data.density import boxes_to_density
from src.utils import read_json
from src.visualization.overlays import draw_boxes, draw_count_banner, overlay_density, overlay_masks

log = logging.getLogger("fruitcount")


def render_all_modalities(
    image: np.ndarray,
    mask: np.ndarray | None,
    boxes: np.ndarray,
    title_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tạo 3 góc nhìn: Box, Mask, và Density map."""
    h, w = image.shape[:2]
    count = len(boxes)

    # 1. Box overlay
    box_img = draw_boxes(image, boxes, color=(46, 204, 113), thickness=max(1, int(round(min(h, w) / 450))))
    box_img = draw_count_banner(box_img, f"{title_prefix} Boxes: {count}")

    # 2. Mask overlay
    if mask is not None and count > 0:
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]
        binary_masks = np.stack([mask == uid for uid in unique_ids])
        mask_img = overlay_masks(image, binary_masks, alpha=0.45)
    else:
        mask_img = image.copy()
    mask_img = draw_count_banner(mask_img, f"{title_prefix} Masks: {count}")

    # 3. Density heatmap
    dens = boxes_to_density(boxes, (h, w), downsample=8, sigma_scale=0.25)
    dens_img = overlay_density(image, dens, alpha=0.55)
    dens_img = draw_count_banner(dens_img, f"{title_prefix} Density Sum: {dens.sum():.1f}")

    return box_img, mask_img, dens_img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aug-root", default=str(ROOT / "data/augmented"))
    ap.add_argument("--out", default=str(ROOT / "runs/augmentation_showcase"))
    ap.add_argument("--num-samples", type=int, default=4)
    args = ap.parse_args()
    setup_logging()

    aug_root = Path(args.aug_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = [d.name for d in aug_root.iterdir() if d.is_dir() and (d / "records.json").exists()]
    if not strategies:
        raise SystemExit(
            f"Không tìm thấy dataset nào trong {aug_root}. Hãy chạy scripts/augment_dataset.py trước!"
        )

    log.info("Tìm thấy %d chiến lược tăng cường: %s", len(strategies), ", ".join(strategies))

    for strat in strategies:
        strat_dir = aug_root / strat
        records = read_json(strat_dir / "records.json")
        # Lọc các ảnh được sinh mới bởi augmentation
        aug_recs = [r for r in records if r["image_id"].startswith("aug_")]
        if not aug_recs:
            aug_recs = records

        strat_out = out_dir / strat
        strat_out.mkdir(parents=True, exist_ok=True)

        samples_to_show = aug_recs[: min(args.num_samples, len(aug_recs))]
        log.info("Đang xuất %d ảnh mẫu trực quan cho chiến lược: %s...", len(samples_to_show), strat)

        for i, r in enumerate(samples_to_show):
            img_path = r["image_path"]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = None
            if r.get("mask_path") and Path(r["mask_path"]).exists():
                mask = read_instance_mask(r["mask_path"])

            boxes = np.asarray(r.get("boxes", []), dtype=np.float32)

            box_img, mask_img, dens_img = render_all_modalities(
                img, mask, boxes, title_prefix=f"[{strat}]"
            )

            # Ghép thành 1 panel so sánh ngang 3 khung hình: Box | Mask | Density
            # Resize về chiều cao đồng nhất 600px
            target_h = 600
            scale = target_h / box_img.shape[0]
            new_w = int(box_img.shape[1] * scale)

            b_res = cv2.resize(box_img, (new_w, target_h))
            m_res = cv2.resize(mask_img, (new_w, target_h))
            d_res = cv2.resize(dens_img, (new_w, target_h))

            triptych = np.hstack([b_res, m_res, d_res])
            triptych_bgr = cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR)

            out_file = strat_out / f"showcase_{i+1:02d}_{r['image_id']}.jpg"
            cv2.imwrite(str(out_file), triptych_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    log.info("=" * 70)
    log.info("Hoàn tất trực quan hóa! Toàn bộ ảnh showcase đã lưu tại: %s", out_dir)


if __name__ == "__main__":
    main()
