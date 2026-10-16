"""Tạo các bộ dữ liệu tăng cường (Augmented Datasets) với ưu điểm riêng cho bài toán FruitCount.

Hỗ trợ 4 chiến lược tăng cường chuyên biệt:
1. dense_occlusion: Chuyên xử lý quả mọc thành chùm dày đặc, chồng lấn và bị lá/cành che khuất.
   -> Ưu điểm: Phục vụ so sánh cốt lõi của đề tài (Object Detection vs Density Regression khi quả chồng lấn).
2. lighting_weather: Chuyên xử lý biến động ánh sáng ngoài trời, bóng râm tán lá rậm rạp, nắng chói lóa.
   -> Ưu điểm: Tăng độ bền vững (robustness) trong điều kiện thực địa nông nghiệp.
3. multiscale_tiling: Chuyên xử lý quả nhỏ ở khoảng cách xa, cắt patch độ phân giải cao.
   -> Ưu điểm: Tối ưu cho cơ chế suy luận SAHI/tiling (--tile) và tránh mất chi tiết quả nhỏ.
4. balanced_hybrid: Kết hợp hài hòa hình học và quang sai để huấn luyện benchmark tổng quát.
5. all: Tự động sinh cả 4 bộ dữ liệu trên.

Cách dùng:
    python scripts/augment_dataset.py --strategy dense_occlusion --multiplier 1.0
    python scripts/augment_dataset.py --strategy all --multiplier 0.5
    python scripts/augment_dataset.py --images-dir d:/CV/BTL/Data/detection/detection/train/images --masks-dir d:/CV/BTL/Data/detection/detection/train/masks
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from _common import ROOT, setup_logging
from src.data.advanced_augmentation import (
    AdaptiveCLAHE,
    CanopyShadow,
    ClusterCopyPaste,
    FarmWeatherSim,
    FoliageOcclusion,
    HighResFruitTileCrop,
    MicroFruitDownscaleSim,
    SunFlare,
    create_mosaic4,
)
from src.data.annotations import (
    ImageRecord,
    load_from_instance_masks,
    masks_to_boxes,
    read_instance_mask,
    save_records,
)
from src.data.augmentation import (
    ColorJitter,
    Compose,
    GaussianBlur,
    HFlip,
    RandomScaleCrop,
    Sample,
    VFlip,
)
from src.data.splits import make_splits, save_splits, select_records
from src.data.yolo_export import export_yolo_dataset
from src.utils import load_yaml, save_yaml

log = logging.getLogger("fruitcount")

STRATEGY_DESCRIPTIONS = {
    "dense_occlusion": {
        "title": "Chùm quả dày đặc & Che khuất cực độ (Dense & Occlusion)",
        "advantage": "Tối ưu hóa khả năng tách quả chồng lấn cho YOLO/Faster R-CNN và thử thách đếm vùng cực đông cho CSRNet.",
        "techniques": ["Foliage/Branch Occlusion", "Cluster Copy-Paste", "Mosaic 4-Image"]
    },
    "lighting_weather": {
        "title": "Độ bền vững Ánh sáng & Môi trường nông trại (Lighting & Farm Weather)",
        "advantage": "Khắc phục hiện tượng bóng râm tán lá che khuất và lóa sáng nắng gắt (sun flare), giúp nhận diện chính xác quả ở mọi thời điểm trong ngày.",
        "techniques": ["Canopy Soft Shadows", "Sun Flare & Overexposure", "Adaptive CLAHE", "Farm Weather Sim (Fog/Dusk)"]
    },
    "multiscale_tiling": {
        "title": "Quả nhỏ & Cắt mảng chi tiết cao (Multi-scale & High-Res Tiling)",
        "advantage": "Bảo tồn 100% độ phân giải cho các quả nhỏ ở xa, tương thích hoàn hảo với cơ chế SAHI/tiling inference.",
        "techniques": ["High-Res Fruit Tile Crop", "Micro-Fruit Downscale Simulation", "Scale Jittering"]
    },
    "balanced_hybrid": {
        "title": "Cân bằng toàn diện (Balanced Hybrid Benchmark)",
        "advantage": "Bộ dữ liệu tổng hòa các biến đổi tự nhiên, độ ổn định cao nhất để huấn luyện benchmark công bằng cho cả 4 mô hình.",
        "techniques": ["Horizontal & Vertical Flip", "Color Jitter", "Adaptive CLAHE", "Light Foliage Occlusion", "Random Scale Crop"]
    }
}


def build_pipeline_for_strategy(strategy: str) -> Compose:
    """Tạo chuỗi biến đổi hình ảnh theo chiến lược tăng cường."""
    if strategy == "dense_occlusion":
        return Compose([
            HFlip(p=0.5),
            FoliageOcclusion(p=0.7, max_occlusions=6),
            ClusterCopyPaste(p=0.5, max_paste_fruits=12),
        ])
    elif strategy == "lighting_weather":
        return Compose([
            HFlip(p=0.5),
            CanopyShadow(p=0.6, min_shadow_factor=0.4, max_shadow_factor=0.75),
            SunFlare(p=0.4, max_intensity=0.7),
            AdaptiveCLAHE(p=0.5, clip_limit=2.5),
            FarmWeatherSim(p=0.4),
        ])
    elif strategy == "multiscale_tiling":
        return Compose([
            HFlip(p=0.5),
            MicroFruitDownscaleSim(p=0.4, downscale_factor_range=(0.4, 0.7)),
            HighResFruitTileCrop(tile_size=640, min_fruits=2, min_visibility=0.35),
        ])
    elif strategy == "balanced_hybrid":
        return Compose([
            HFlip(p=0.5),
            ColorJitter(p=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=8),
            AdaptiveCLAHE(p=0.3, clip_limit=2.0),
            FoliageOcclusion(p=0.3, max_occlusions=3),
            RandomScaleCrop(p=0.5, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        ])
    else:
        raise ValueError(f"Chiến lược không hợp lệ: {strategy}")


def augment_single_record(
    record: ImageRecord,
    strategy: str,
    pipeline: Compose,
    new_id: str,
    out_img_dir: Path,
    out_mask_dir: Path,
    rng: np.random.Generator,
) -> Optional[ImageRecord]:
    """Tạo 1 mẫu tăng cường từ ImageRecord gốc và lưu ảnh + mask tương ứng."""
    img = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inst = None
    if record.mask_path and Path(record.mask_path).exists():
        inst = read_instance_mask(record.mask_path)

    sample = Sample(img, record.boxes.copy(), record.instance_ids.copy(), inst)
    aug_sample = pipeline(sample)

    # Đảm bảo box hợp lệ
    boxes = aug_sample.boxes
    ids = aug_sample.ids
    mask = aug_sample.inst_mask

    if len(boxes) == 0:
        return None

    h, w = aug_sample.height, aug_sample.width
    # Lưu ảnh RGB
    out_img_path = out_img_dir / f"{new_id}.png"
    cv2.imwrite(str(out_img_path), cv2.cvtColor(aug_sample.image, cv2.COLOR_RGB2BGR))

    # Lưu mask nếu có
    out_mask_path = None
    if mask is not None:
        out_mask_path = out_mask_dir / f"{new_id}.png"
        if mask.max() > 255:
            cv2.imwrite(str(out_mask_path), mask.astype(np.uint16))
        else:
            cv2.imwrite(str(out_mask_path), mask.astype(np.uint8))

    return ImageRecord(
        image_id=new_id,
        image_path=str(out_img_path),
        width=w,
        height=h,
        boxes=boxes,
        instance_ids=ids,
        mask_path=str(out_mask_path) if out_mask_path else None,
    )


def generate_augmented_dataset(
    base_records: List[ImageRecord],
    strategy: str,
    out_root: Path,
    multiplier: float = 1.0,
    seed: int = 42,
    export_yolo: bool = True,
) -> Path:
    """Sinh toàn bộ tập dữ liệu tăng cường cho 1 chiến lược cụ thể."""
    rng = np.random.default_rng(seed)
    info = STRATEGY_DESCRIPTIONS[strategy]
    log.info("=" * 70)
    log.info("Bắt đầu sinh dữ liệu: %s", info["title"])
    log.info("Ưu điểm riêng: %s", info["advantage"])
    log.info("Kỹ thuật: %s", ", ".join(info["techniques"]))

    ds_dir = out_root / strategy
    images_dir = ds_dir / "images"
    masks_dir = ds_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline_for_strategy(strategy)

    # 1. Giữ các ảnh gốc (bản copy hoặc symlink vào dataset)
    augmented_records: List[ImageRecord] = []
    log.info("Sao chép dữ liệu cơ sở vào tập đích...")
    for r in base_records:
        orig_img_dst = images_dir / f"{r.image_id}.png"
        if not orig_img_dst.exists():
            shutil.copy2(r.image_path, orig_img_dst)

        orig_mask_dst = None
        if r.mask_path and Path(r.mask_path).exists():
            orig_mask_dst = masks_dir / f"{r.image_id}.png"
            if not orig_mask_dst.exists():
                shutil.copy2(r.mask_path, orig_mask_dst)

        augmented_records.append(
            ImageRecord(
                image_id=r.image_id,
                image_path=str(orig_img_dst),
                width=r.width,
                height=r.height,
                boxes=r.boxes.copy(),
                instance_ids=r.instance_ids.copy(),
                mask_path=str(orig_mask_dst) if orig_mask_dst else None,
            )
        )

    # 2. Sinh các mẫu tăng cường theo số lượng mong muốn
    num_to_generate = int(round(len(base_records) * multiplier))
    log.info("Đang sinh thêm %d mẫu tăng cường...", num_to_generate)

    # Nếu là dense_occlusion: xen kẽ thêm Mosaic-4
    mosaic_count = 0
    if strategy == "dense_occlusion" and len(base_records) >= 4:
        num_mosaics = int(num_to_generate * 0.25)
        for m_idx in range(num_mosaics):
            chosen = rng.choice(base_records, size=4, replace=False)
            samples_4 = []
            valid = True
            for cr in chosen:
                c_img = cv2.imread(cr.image_path, cv2.IMREAD_COLOR)
                if c_img is None:
                    valid = False
                    break
                c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
                c_mask = read_instance_mask(cr.mask_path) if cr.mask_path else None
                samples_4.append(Sample(c_img, cr.boxes.copy(), cr.instance_ids.copy(), c_mask))
            if not valid:
                continue

            mosaic_sample = create_mosaic4(samples_4, target_size=(1024, 1024), rng=rng)
            if len(mosaic_sample.boxes) == 0:
                continue

            mid = f"aug_mosaic_{m_idx:04d}"
            m_img_path = images_dir / f"{mid}.png"
            m_mask_path = masks_dir / f"{mid}.png"
            cv2.imwrite(str(m_img_path), cv2.cvtColor(mosaic_sample.image, cv2.COLOR_RGB2BGR))
            if mosaic_sample.inst_mask is not None:
                if mosaic_sample.inst_mask.max() > 255:
                    cv2.imwrite(str(m_mask_path), mosaic_sample.inst_mask.astype(np.uint16))
                else:
                    cv2.imwrite(str(m_mask_path), mosaic_sample.inst_mask.astype(np.uint8))

            augmented_records.append(
                ImageRecord(
                    image_id=mid,
                    image_path=str(m_img_path),
                    width=1024,
                    height=1024,
                    boxes=mosaic_sample.boxes,
                    instance_ids=mosaic_sample.ids,
                    mask_path=str(m_mask_path),
                )
            )
            mosaic_count += 1

    remaining_to_gen = num_to_generate - mosaic_count
    for i in range(remaining_to_gen):
        rec_idx = i % len(base_records)
        base_rec = base_records[rec_idx]
        new_id = f"aug_{strategy}_{i:04d}"
        aug_rec = augment_single_record(
            base_rec,
            strategy,
            pipeline,
            new_id,
            images_dir,
            masks_dir,
            rng,
        )
        if aug_rec is not None:
            augmented_records.append(aug_rec)

    # 3. Chia tập train / val / test (stratified theo số quả)
    splits = make_splits(augmented_records, val=0.15, test=0.15, seed=seed, stratify_by_count=True)
    save_records(augmented_records, ds_dir / "records.json")
    save_splits(splits, ds_dir / "splits.json")

    # 4. Xuất dataset định dạng Ultralytics YOLO
    if export_yolo:
        by_split = {k: select_records(augmented_records, v) for k, v in splits.items()}
        export_yolo_dataset(by_split, ds_dir / "yolo", class_names=["fruit"])
        log.info("Đã xuất định dạng YOLO tại: %s", ds_dir / "yolo")

    # 5. Lưu tóm tắt metadata và ưu điểm
    counts = [r.count for r in augmented_records]
    summary = {
        "strategy": strategy,
        "title": info["title"],
        "advantage": info["advantage"],
        "techniques": info["techniques"],
        "total_images": len(augmented_records),
        "total_fruits": int(sum(counts)),
        "fruit_stats": {
            "min": int(min(counts)) if counts else 0,
            "mean": float(round(sum(counts) / max(1, len(counts)), 2)),
            "max": int(max(counts)) if counts else 0,
        },
        "splits": {k: len(v) for k, v in splits.items()}
    }
    with open(ds_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 6. Tạo file cấu hình dataset YAML tương ứng để train ngay lập tức
    configs_dir = ROOT / "configs"
    cfg_file = configs_dir / f"dataset_{strategy}.yaml"
    cfg_content = {
        "name": f"minneapple_{strategy}",
        "description": f"{info['title']} - {info['advantage']}",
        "format": "instance_masks",
        "paths": {
            "images_dir": str(images_dir.relative_to(ROOT)).replace("\\", "/"),
            "masks_dir": str(masks_dir.relative_to(ROOT)).replace("\\", "/"),
            "mask_suffix": "",
            "labels_dir": None,
            "coco_json": None,
        },
        "min_instance_area": 15,
        "split": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
            "seed": seed,
            "stratify_by_count": True,
        },
        "processed_dir": str(ds_dir.relative_to(ROOT)).replace("\\", "/"),
        "yolo_export": {
            "symlink": False,
        },
        "classes": ["fruit"],
    }
    save_yaml(cfg_content, cfg_file)
    log.info("Đã sinh config dataset: %s", cfg_file)
    log.info(
        "Hoàn tất %s: Tổng %d ảnh (Gốc: %d, Sinh mới: %d) | Quả/ảnh TB: %.1f",
        strategy, len(augmented_records), len(base_records), len(augmented_records) - len(base_records), summary["fruit_stats"]["mean"]
    )
    return ds_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--source-dataset",
        default=str(ROOT / "configs/dataset.yaml"),
        help="Đường dẫn file dataset.yaml chứa cấu hình nguồn"
    )
    ap.add_argument(
        "--images-dir",
        default="d:/CV/BTL/Data/detection/detection/train/images",
        help="Thư mục ảnh gốc (ưu tiên hơn config nếu tồn tại)"
    )
    ap.add_argument(
        "--masks-dir",
        default="d:/CV/BTL/Data/detection/detection/train/masks",
        help="Thư mục mask gốc (ưu tiên hơn config nếu tồn tại)"
    )
    ap.add_argument(
        "--strategy",
        choices=["all", "dense_occlusion", "lighting_weather", "multiscale_tiling", "balanced_hybrid"],
        default="all",
        help="Chiến lược tăng cường dữ liệu"
    )
    ap.add_argument(
        "--multiplier",
        type=float,
        default=0.5,
        help="Tỉ lệ số ảnh sinh thêm so với tập gốc (ví dụ 0.5 = sinh thêm 50% ảnh mới)"
    )
    ap.add_argument(
        "--max-base-samples",
        type=int,
        default=None,
        help="Giới hạn số ảnh gốc đọc vào (dùng để test nhanh pipeline)"
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "data/augmented"),
        help="Thư mục lưu các dataset tăng cường"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    setup_logging()

    # Nạp dữ liệu gốc
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)

    if not images_dir.exists():
        # Fallback đọc từ config
        log.info("Không tìm thấy %s, đọc từ %s...", images_dir, args.source_dataset)
        cfg = load_yaml(args.source_dataset)
        p = cfg["paths"]
        images_dir = ROOT / p["images_dir"]
        masks_dir = ROOT / p["masks_dir"]

    log.info("Đang đọc dữ liệu gốc từ:")
    log.info("  Images: %s", images_dir)
    log.info("  Masks:  %s", masks_dir)

    base_records = load_from_instance_masks(images_dir, masks_dir, mask_suffix="", min_area=20)
    if not base_records:
        raise SystemExit(f"Không tìm thấy ảnh hoặc mask tại {images_dir} và {masks_dir}!")

    log.info("Đã nạp thành công %d ảnh gốc có nhãn.", len(base_records))
    if args.max_base_samples and args.max_base_samples < len(base_records):
        rng_sub = np.random.default_rng(args.seed)
        sub_indices = rng_sub.choice(len(base_records), size=args.max_base_samples, replace=False)
        base_records = [base_records[i] for i in sorted(sub_indices)]
        log.info("Giới hạn xử lý %d ảnh gốc theo tham số --max-base-samples.", len(base_records))

    strategies = (
        ["dense_occlusion", "lighting_weather", "multiscale_tiling", "balanced_hybrid"]
        if args.strategy == "all"
        else [args.strategy]
    )

    out_root = Path(args.out)
    for strat in strategies:
        generate_augmented_dataset(
            base_records=base_records,
            strategy=strat,
            out_root=out_root,
            multiplier=args.multiplier,
            seed=args.seed,
            export_yolo=True,
        )

    log.info("=" * 70)
    log.info("TẤT CẢ CÁC BỘ DỮ LIỆU TĂNG CƯỜNG ĐÃ ĐƯỢC TẠO THÀNH CÔNG TẠI: %s", out_root)
    log.info("Các file config đã sinh sẵn:")
    for strat in strategies:
        log.info("  - configs/dataset_%s.yaml", strat)


if __name__ == "__main__":
    main()
