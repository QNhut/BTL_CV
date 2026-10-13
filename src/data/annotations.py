"""Đọc annotation từ nhiều định dạng và quy về một cấu trúc thống nhất.

Mọi model (YOLO, Faster R-CNN, Mask R-CNN, CSRNet) đều dùng chung `ImageRecord`
nên chỉ cần một bước tiền xử lý. Box là nguồn sự thật; density map và nhãn
YOLO đều suy ra từ box.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from ..utils.io import read_json, write_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageRecord:
    image_id: str
    image_path: str
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # xyxy, pixel
    instance_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    mask_path: Optional[str] = None  # ảnh instance mask (nếu có)

    @property
    def count(self) -> int:
        return int(len(self.boxes))

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "boxes": np.asarray(self.boxes, dtype=np.float32).round(2).tolist(),
            "instance_ids": np.asarray(self.instance_ids, dtype=np.int64).tolist(),
            "mask_path": self.mask_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageRecord":
        boxes = np.asarray(d.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        ids = d.get("instance_ids")
        ids = (
            np.asarray(ids, dtype=np.int64)
            if ids is not None and len(ids) == len(boxes)
            else np.arange(1, len(boxes) + 1, dtype=np.int64)
        )
        return cls(
            image_id=d["image_id"],
            image_path=d["image_path"],
            width=int(d["width"]),
            height=int(d["height"]),
            boxes=boxes,
            instance_ids=ids,
            mask_path=d.get("mask_path"),
        )


# --------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------
def list_images(images_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_image_size(path: str | Path) -> tuple[int, int]:
    """Trả về (width, height) mà không cần giữ ảnh trong bộ nhớ lâu."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    h, w = img.shape[:2]
    return w, h


def read_instance_mask(path: str | Path) -> np.ndarray:
    """Đọc ảnh instance mask thành mảng int32 (H, W); 0 = nền.

    Hỗ trợ mask 1 kênh (id trong giá trị pixel) và mask 3 kênh (id = R*65536+G*256+B).
    """
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    if m.ndim == 3:
        b, g, r = m[..., 0].astype(np.int64), m[..., 1].astype(np.int64), m[..., 2].astype(np.int64)
        m = r * 65536 + g * 256 + b
    return m.astype(np.int32)


def masks_to_boxes(
    instance_mask: np.ndarray, min_area: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Suy box xyxy từ instance mask. Trả về (boxes[N,4], ids[N])."""
    ids = np.unique(instance_mask)
    ids = ids[ids > 0]
    boxes, kept = [], []
    for i in ids:
        ys, xs = np.nonzero(instance_mask == i)
        if len(xs) < min_area:
            continue
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        kept.append(i)
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    return np.asarray(boxes, np.float32), np.asarray(kept, np.int64)


# --------------------------------------------------------------------------
# Loader theo định dạng
# --------------------------------------------------------------------------
def load_from_instance_masks(
    images_dir: str | Path,
    masks_dir: str | Path,
    mask_suffix: str = "",
    min_area: int = 20,
) -> list[ImageRecord]:
    """MinneApple và các bộ dữ liệu có instance mask: mỗi quả một id."""
    records: list[ImageRecord] = []
    masks_dir = Path(masks_dir)
    for img_path in list_images(images_dir):
        mask_path = None
        for ext in IMAGE_EXTS:
            cand = masks_dir / f"{img_path.stem}{mask_suffix}{ext}"
            if cand.exists():
                mask_path = cand
                break
        if mask_path is None:
            continue  # bỏ qua ảnh không có nhãn
        inst = read_instance_mask(mask_path)
        boxes, ids = masks_to_boxes(inst, min_area=min_area)
        h, w = inst.shape[:2]
        records.append(
            ImageRecord(
                image_id=img_path.stem,
                image_path=str(img_path),
                width=w,
                height=h,
                boxes=boxes,
                instance_ids=ids,
                mask_path=str(mask_path),
            )
        )
    return records


def load_from_yolo_txt(images_dir: str | Path, labels_dir: str | Path) -> list[ImageRecord]:
    """Nhãn YOLO: mỗi dòng `cls cx cy w h` (chuẩn hóa 0-1)."""
    records: list[ImageRecord] = []
    labels_dir = Path(labels_dir)
    for img_path in list_images(images_dir):
        w, h = read_image_size(img_path)
        lab = labels_dir / f"{img_path.stem}.txt"
        boxes = []
        if lab.exists():
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(x) for x in parts[1:5])
                boxes.append(
                    [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                )
        arr = np.asarray(boxes, np.float32).reshape(-1, 4)
        records.append(
            ImageRecord(
                image_id=img_path.stem,
                image_path=str(img_path),
                width=w,
                height=h,
                boxes=arr,
                instance_ids=np.arange(1, len(arr) + 1, dtype=np.int64),
            )
        )
    return records


def load_from_coco(coco_json: str | Path, images_dir: str | Path) -> list[ImageRecord]:
    """COCO detection: bbox = [x, y, w, h]; bỏ qua annotation iscrowd."""
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    per_image: dict[int, list] = {}
    for a in data.get("annotations", []):
        if a.get("iscrowd", 0):
            continue
        per_image.setdefault(a["image_id"], []).append(a["bbox"])
    records = []
    for im in data["images"]:
        path = Path(images_dir) / im["file_name"]
        bbs = per_image.get(im["id"], [])
        arr = np.asarray([[x, y, x + w, y + h] for x, y, w, h in bbs], np.float32).reshape(-1, 4)
        records.append(
            ImageRecord(
                image_id=Path(im["file_name"]).stem,
                image_path=str(path),
                width=int(im["width"]),
                height=int(im["height"]),
                boxes=arr,
                instance_ids=np.arange(1, len(arr) + 1, dtype=np.int64),
            )
        )
    return records


def load_records_from_config(cfg: dict) -> list[ImageRecord]:
    """Dựng danh sách `ImageRecord` từ configs/dataset.yaml."""
    fmt = cfg.get("format", "instance_masks")
    p = cfg["paths"]
    if fmt == "instance_masks":
        return load_from_instance_masks(
            p["images_dir"], p["masks_dir"], p.get("mask_suffix", ""), cfg.get("min_instance_area", 20)
        )
    if fmt == "yolo_txt":
        return load_from_yolo_txt(p["images_dir"], p["labels_dir"])
    if fmt == "coco":
        return load_from_coco(p["coco_json"], p["images_dir"])
    raise ValueError(f"Định dạng annotation không hỗ trợ: {fmt}")


# --------------------------------------------------------------------------
# Lưu / nạp records
# --------------------------------------------------------------------------
def save_records(records: Iterable[ImageRecord], path: str | Path) -> None:
    write_json([r.to_dict() for r in records], path)


def load_records(path: str | Path) -> list[ImageRecord]:
    return [ImageRecord.from_dict(d) for d in read_json(path)]
