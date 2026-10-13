"""PyTorch Dataset cho từng họ model, cùng đọc `ImageRecord` + `Compose` augmenter.

- detection    : Faster R-CNN (box)
- segmentation : Mask R-CNN (box + mask từng quả)
- density      : CSRNet (ảnh + density map)
YOLO không dùng các lớp này; nó đọc dataset đã xuất bởi `yolo_export.py`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import load_image
from .annotations import ImageRecord, read_instance_mask
from .augmentation import Compose, Sample
from .density import boxes_to_density


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (3, H, W) float32 trong [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)


class _FruitDataset(Dataset):
    need_mask = False

    def __init__(self, records: list[ImageRecord], augmenter: Optional[Compose] = None):
        self.records = records
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.records)

    def _sample(self, idx: int) -> Sample:
        r = self.records[idx]
        img = load_image(r.image_path)
        inst = None
        if self.need_mask:
            if not r.mask_path:
                raise ValueError(
                    f"Ảnh {r.image_id} không có instance mask; Mask R-CNN cần annotation dạng instance_masks"
                )
            inst = read_instance_mask(r.mask_path)
        s = Sample(img, r.boxes.copy(), r.instance_ids.copy(), inst)
        return self.augmenter(s) if self.augmenter else s


class FruitDetectionDataset(_FruitDataset):
    """Trả (image[3,H,W], target) theo chuẩn torchvision detection."""

    def __init__(self, records, augmenter=None, with_masks: bool = False):
        super().__init__(records, augmenter)
        self.need_mask = with_masks

    def __getitem__(self, idx: int):
        s = self._sample(idx)
        n = len(s.boxes)
        target = {
            "boxes": torch.as_tensor(s.boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((n,), dtype=torch.int64),  # lớp 1 = quả (0 là nền)
            "image_id": torch.tensor(idx),
            "area": torch.as_tensor(
                (s.boxes[:, 2] - s.boxes[:, 0]) * (s.boxes[:, 3] - s.boxes[:, 1]), dtype=torch.float32
            ) if n else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((n,), dtype=torch.int64),
        }
        if self.need_mask:
            h, w = s.image.shape[:2]
            if n:
                masks = np.stack([s.inst_mask == i for i in s.ids]).astype(np.uint8)
            else:
                masks = np.zeros((0, h, w), np.uint8)
            target["masks"] = torch.from_numpy(masks)
        return image_to_tensor(s.image), target


class FruitDensityDataset(_FruitDataset):
    """Trả (image[3,H,W], density[1,H/ds,W/ds], count)."""

    def __init__(self, records, augmenter=None, density_cfg: Optional[dict] = None):
        super().__init__(records, augmenter)
        self.cfg = {"downsample": 8, "sigma_scale": 0.25, "min_sigma": 1.0, "fixed_sigma": None}
        self.cfg.update(density_cfg or {})

    def __getitem__(self, idx: int):
        s = self._sample(idx)
        dens = boxes_to_density(s.boxes, s.image.shape[:2], **self.cfg)
        return (
            image_to_tensor(s.image),
            torch.from_numpy(dens)[None],
            torch.tensor(float(len(s.boxes))),
        )


def detection_collate(batch):
    return tuple(zip(*batch))


def density_collate(batch):
    """Pad ảnh / density về cùng kích thước lớn nhất trong batch (pad 0)."""
    imgs, dens, counts = zip(*batch)
    ds = imgs[0].shape[-1] // max(1, dens[0].shape[-1])
    H = max(i.shape[-2] for i in imgs)
    W = max(i.shape[-1] for i in imgs)
    H, W = -(-H // ds) * ds, -(-W // ds) * ds
    out_i = torch.zeros(len(imgs), 3, H, W)
    out_d = torch.zeros(len(imgs), 1, H // ds, W // ds)
    for k, (i, d) in enumerate(zip(imgs, dens)):
        out_i[k, :, : i.shape[-2], : i.shape[-1]] = i
        out_d[k, :, : d.shape[-2], : d.shape[-1]] = d
    return out_i, out_d, torch.stack(counts)


def build_dataset(
    kind: str, records: list[ImageRecord], augmenter: Optional[Compose] = None,
    density_cfg: Optional[dict] = None,
) -> Dataset:
    if kind == "detection":
        return FruitDetectionDataset(records, augmenter, with_masks=False)
    if kind == "segmentation":
        return FruitDetectionDataset(records, augmenter, with_masks=True)
    if kind == "density":
        return FruitDensityDataset(records, augmenter, density_cfg)
    raise ValueError(f"Loại dataset không hỗ trợ: {kind}")


def get_collate_fn(kind: str):
    return density_collate if kind == "density" else detection_collate
