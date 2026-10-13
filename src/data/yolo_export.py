"""Xuất dataset sang định dạng Ultralytics:

    <out>/images/{train,val,test}/*.jpg
    <out>/labels/{train,val,test}/*.txt   (cls cx cy w h, chuẩn hóa)
    <out>/data.yaml
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from ..utils.config import save_yaml
from .annotations import ImageRecord


def boxes_to_yolo_lines(boxes: np.ndarray, width: int, height: int, cls: int = 0) -> list[str]:
    lines = []
    for x1, y1, x2, y2 in np.asarray(boxes, np.float32).reshape(-1, 4):
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        lines.append(
            f"{cls} {(x1 + bw / 2) / width:.6f} {(y1 + bh / 2) / height:.6f} "
            f"{bw / width:.6f} {bh / height:.6f}"
        )
    return lines


def export_yolo_dataset(
    records_by_split: dict[str, list[ImageRecord]],
    out_dir: str | Path,
    class_names: list[str] | None = None,
    symlink: bool = False,
) -> Path:
    """Ghi ảnh + nhãn từng split và trả về đường dẫn data.yaml."""
    out_dir = Path(out_dir).resolve()
    for split, recs in records_by_split.items():
        img_dir = out_dir / "images" / split
        lab_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for r in recs:
            src = Path(r.image_path).resolve()
            dst = img_dir / f"{r.image_id}{src.suffix.lower()}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if symlink:
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            (lab_dir / f"{r.image_id}.txt").write_text(
                "\n".join(boxes_to_yolo_lines(r.boxes, r.width, r.height)) + "\n"
                if len(r.boxes)
                else ""
            )
    names = class_names or ["fruit"]
    data = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    if "test" in records_by_split:
        data["test"] = "images/test"
    yaml_path = out_dir / "data.yaml"
    save_yaml(data, yaml_path)
    return yaml_path
