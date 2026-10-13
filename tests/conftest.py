"""Fixture dùng chung: dataset giả (ảnh vườn tổng hợp + instance mask) để test không cần dữ liệu thật."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_synthetic_dataset(root: Path, n: int = 24, size=(192, 256), seed: int = 0):
    """Tạo n ảnh nền xanh với các "quả" tròn đỏ không chồng nhau.

    Trả về (images_dir, masks_dir, {image_id: số quả}). Mask là PNG uint16, mỗi quả một id.
    """
    rng = np.random.default_rng(seed)
    images_dir, masks_dir = root / "images", root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    h, w = size
    counts: dict[str, int] = {}
    for k in range(n):
        img = np.full((h, w, 3), (40, 120, 40), np.uint8)
        img += rng.integers(0, 20, img.shape, dtype=np.uint8)
        mask = np.zeros((h, w), np.uint16)
        target = int(rng.integers(1, 13))
        centers: list[tuple[int, int, int]] = []
        tries = 0
        while len(centers) < target and tries < 500:
            tries += 1
            r = int(rng.integers(6, 12))
            cx, cy = int(rng.integers(r + 1, w - r - 1)), int(rng.integers(r + 1, h - r - 1))
            if all((cx - x) ** 2 + (cy - y) ** 2 > (r + rr + 2) ** 2 for x, y, rr in centers):
                centers.append((cx, cy, r))
        for i, (cx, cy, r) in enumerate(centers, start=1):
            cv2.circle(img, (cx, cy), r, (200, 30, 30), -1)
            cv2.circle(mask, (cx, cy), r, i, -1)
        name = f"img_{k:03d}"
        cv2.imwrite(str(images_dir / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(masks_dir / f"{name}.png"), mask)
        counts[name] = len(centers)
    return images_dir, masks_dir, counts


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    return make_synthetic_dataset(root)


@pytest.fixture(scope="session")
def records(synthetic):
    from src.data import load_from_instance_masks

    images_dir, masks_dir, _ = synthetic
    return load_from_instance_masks(images_dir, masks_dir, min_area=10)
