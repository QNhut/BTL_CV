"""Augmentation bằng numpy + OpenCV, biến đổi đồng bộ ảnh / box / instance mask.

Dùng cho Faster R-CNN, Mask R-CNN và CSRNet (density map được sinh *sau* augmentation
từ box đã biến đổi, nên luôn khớp với ảnh). YOLO dùng augmentation riêng của Ultralytics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass
class Sample:
    image: np.ndarray                       # (H, W, 3) uint8 RGB
    boxes: np.ndarray                       # (N, 4) float32 xyxy
    ids: np.ndarray                         # (N,) id instance (khớp giá trị trong inst_mask)
    inst_mask: Optional[np.ndarray] = None  # (H, W) int32, 0 = nền

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


# --------------------------------------------------------------------------
# Biến đổi cơ bản
# --------------------------------------------------------------------------
class Transform:
    def __call__(self, s: Sample, rng: np.random.Generator) -> Sample:  # pragma: no cover
        raise NotImplementedError


class HFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, s, rng):
        if rng.random() >= self.p:
            return s
        w = s.width
        b = s.boxes.copy()
        if len(b):
            b[:, [0, 2]] = w - s.boxes[:, [2, 0]]
        m = None if s.inst_mask is None else np.ascontiguousarray(s.inst_mask[:, ::-1])
        return Sample(np.ascontiguousarray(s.image[:, ::-1]), b, s.ids, m)


class VFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, s, rng):
        if rng.random() >= self.p:
            return s
        h = s.height
        b = s.boxes.copy()
        if len(b):
            b[:, [1, 3]] = h - s.boxes[:, [3, 1]]
        m = None if s.inst_mask is None else np.ascontiguousarray(s.inst_mask[::-1])
        return Sample(np.ascontiguousarray(s.image[::-1]), b, s.ids, m)


class ColorJitter(Transform):
    def __init__(self, p=0.8, brightness=0.25, contrast=0.25, saturation=0.25, hue=10):
        self.p, self.brightness, self.contrast = p, brightness, contrast
        self.saturation, self.hue = saturation, hue

    def __call__(self, s, rng):
        if rng.random() >= self.p:
            return s
        img = s.image.astype(np.float32)
        if self.brightness:
            img *= 1.0 + rng.uniform(-self.brightness, self.brightness)
        if self.contrast:
            c = 1.0 + rng.uniform(-self.contrast, self.contrast)
            img = (img - img.mean()) * c + img.mean()
        img = np.clip(img, 0, 255).astype(np.uint8)
        if self.saturation or self.hue:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            if self.saturation:
                hsv[..., 1] *= 1.0 + rng.uniform(-self.saturation, self.saturation)
            if self.hue:
                hsv[..., 0] = (hsv[..., 0] + rng.uniform(-self.hue, self.hue)) % 180
            hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Sample(img, s.boxes, s.ids, s.inst_mask)


class GaussianBlur(Transform):
    def __init__(self, p=0.1, ksize: Sequence[int] = (3, 5)):
        self.p, self.ksize = p, list(ksize)

    def __call__(self, s, rng):
        if rng.random() >= self.p:
            return s
        k = int(rng.choice(self.ksize))
        k += (k % 2 == 0)  # kernel phải lẻ
        return Sample(cv2.GaussianBlur(s.image, (k, k), 0), s.boxes, s.ids, s.inst_mask)


# --------------------------------------------------------------------------
# Crop / resize / pad
# --------------------------------------------------------------------------
def crop_sample(s: Sample, x0: int, y0: int, x1: int, y1: int, min_visibility: float = 0.35) -> Sample:
    """Cắt vùng [x0:x1, y0:y1]; giữ box còn nhìn thấy >= min_visibility diện tích."""
    img = s.image[y0:y1, x0:x1]
    mask = None if s.inst_mask is None else s.inst_mask[y0:y1, x0:x1]
    b, ids = s.boxes, s.ids
    if len(b):
        area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        shifted = b - np.array([x0, y0, x0, y0], np.float32)
        nb = shifted.copy()
        nb[:, [0, 2]] = np.clip(nb[:, [0, 2]], 0, x1 - x0)
        nb[:, [1, 3]] = np.clip(nb[:, [1, 3]], 0, y1 - y0)
        w, h = nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]
        keep = (w * h / np.maximum(area0, 1e-6) >= min_visibility) & (w >= 2) & (h >= 2)
        if mask is not None:
            # Quả bị cắt ở mép crop: thu box về đúng phần mask còn nhìn thấy (box khớp mask)
            for k in np.nonzero(keep & np.any(nb != shifted, axis=1))[0]:
                bx0, by0 = int(np.floor(nb[k, 0])), int(np.floor(nb[k, 1]))
                bx1, by1 = int(np.ceil(nb[k, 2])), int(np.ceil(nb[k, 3]))
                ys, xs = np.nonzero(mask[by0:by1, bx0:bx1] == ids[k])
                if len(xs) < 2:
                    keep[k] = False
                    continue
                nb[k] = [bx0 + xs.min(), by0 + ys.min(), bx0 + xs.max() + 1, by0 + ys.max() + 1]
            w, h = nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]
            keep &= (w >= 2) & (h >= 2)
        b, ids = nb[keep].astype(np.float32), s.ids[keep]
    return Sample(np.ascontiguousarray(img), b, ids, None if mask is None else np.ascontiguousarray(mask))


class RandomScaleCrop(Transform):
    """Crop ngẫu nhiên theo tỉ lệ diện tích + tỉ lệ khung; thử nhiều lần để giữ lại >= 1 quả."""

    def __init__(self, p=0.7, scale=(0.5, 1.0), ratio=(0.75, 1.33), min_visibility=0.35, tries=10):
        self.p, self.scale, self.ratio = p, tuple(scale), tuple(ratio)
        self.min_visibility, self.tries = min_visibility, tries

    def __call__(self, s, rng):
        if rng.random() >= self.p:
            return s
        h, w = s.height, s.width
        best = None
        for _ in range(self.tries):
            area = rng.uniform(*self.scale) * h * w
            r = math.exp(rng.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            cw = int(round(min(w, math.sqrt(area * r))))
            ch = int(round(min(h, math.sqrt(area / r))))
            cw, ch = min(w, max(cw, 8)), min(h, max(ch, 8))
            x0 = int(rng.integers(0, w - cw + 1))
            y0 = int(rng.integers(0, h - ch + 1))
            out = crop_sample(s, x0, y0, x0 + cw, y0 + ch, self.min_visibility)
            best = best or out
            if len(out.boxes) > 0 or len(s.boxes) == 0:
                return out
        return best


class RandomCropFixed(Transform):
    """Crop kích thước cố định (pad 0 nếu ảnh nhỏ hơn) — dùng khi train density."""

    def __init__(self, size: int = 512, min_visibility: float = 0.35, tries: int = 10):
        self.size, self.min_visibility, self.tries = size, min_visibility, tries

    def __call__(self, s, rng):
        s = PadToMin(self.size)(s, rng)
        h, w = s.height, s.width
        out = s
        for _ in range(self.tries):
            x0 = int(rng.integers(0, w - self.size + 1))
            y0 = int(rng.integers(0, h - self.size + 1))
            out = crop_sample(s, x0, y0, x0 + self.size, y0 + self.size, self.min_visibility)
            if len(out.boxes) > 0 or len(s.boxes) == 0:
                break
        return out


class PadToMin(Transform):
    """Pad phải/dưới bằng 0 để cạnh nhỏ nhất >= size."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, s, rng=None):
        ph, pw = max(0, self.size - s.height), max(0, self.size - s.width)
        return _pad(s, ph, pw)


class PadToMultiple(Transform):
    def __init__(self, multiple: int = 8):
        self.multiple = multiple

    def __call__(self, s, rng=None):
        m = self.multiple
        return _pad(s, (-s.height) % m, (-s.width) % m)


def _pad(s: Sample, ph: int, pw: int) -> Sample:
    if ph == 0 and pw == 0:
        return s
    img = np.pad(s.image, ((0, ph), (0, pw), (0, 0)))
    mask = None if s.inst_mask is None else np.pad(s.inst_mask, ((0, ph), (0, pw)))
    return Sample(img, s.boxes, s.ids, mask)


class ResizeLongest(Transform):
    """Đưa cạnh dài về `max_side`. Mặc định chỉ thu nhỏ (only_shrink=True)."""

    def __init__(self, max_side: int = 1024, only_shrink: bool = True):
        self.max_side, self.only_shrink = max_side, only_shrink

    def __call__(self, s, rng=None):
        if not self.max_side:
            return s
        h, w = s.height, s.width
        scale = self.max_side / max(h, w)
        if (self.only_shrink and scale >= 1.0) or scale == 1.0:
            return s
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(s.image, (nw, nh), interpolation=interp)
        mask = None
        if s.inst_mask is not None:
            mask = cv2.resize(s.inst_mask.astype(np.int32), (nw, nh), interpolation=cv2.INTER_NEAREST)
        b = s.boxes.copy()
        if len(b):
            b[:, [0, 2]] *= nw / w
            b[:, [1, 3]] *= nh / h
        return Sample(img, b.astype(np.float32), s.ids, mask)


# --------------------------------------------------------------------------
# Compose + builder
# --------------------------------------------------------------------------
class Compose:
    def __init__(self, transforms: Sequence[Transform], seed: Optional[int] = None):
        self.transforms = list(transforms)
        self.seed = seed
        self._rng: Optional[np.random.Generator] = None
        self._worker_seed = None

    @property
    def rng(self) -> np.random.Generator:
        import sys
        torch = sys.modules.get("torch")
        worker = torch.utils.data.get_worker_info() if torch is not None else None
        worker_seed = worker.seed if worker is not None else self.seed
        # tạo lười để mỗi worker của DataLoader có luồng ngẫu nhiên riêng (khi seed=None)
        if self._rng is None or self._worker_seed != worker_seed:
            self._rng = np.random.default_rng(worker_seed)
            self._worker_seed = worker_seed
        return self._rng

    def __call__(self, s: Sample) -> Sample:
        for t in self.transforms:
            s = t(s, self.rng)
        return s


def build_augmenter(
    cfg: dict, train: bool = True, seed: Optional[int] = None, crop_size: Optional[int] = None,
    pad_multiple: Optional[int] = None,
) -> Compose:
    """Dựng pipeline từ configs/augmentation.yaml (`cfg` là toàn bộ file).

    crop_size: ép crop cố định (density khi train). pad_multiple: pad về bội số (CSRNet).
    """
    sec = cfg.get("train" if train else "val", {}) or {}
    ts: list[Transform] = []
    if train:
        if crop_size:
            # thu nhỏ về cùng tỉ lệ với lúc suy luận rồi mới crop
            if sec.get("max_side"):
                ts.append(ResizeLongest(int(sec["max_side"])))
            ts.append(RandomCropFixed(crop_size))
        elif "random_scale_crop" in sec:
            ts.append(RandomScaleCrop(**sec["random_scale_crop"]))
        if sec.get("hflip"):
            ts.append(HFlip(sec["hflip"]))
        if sec.get("vflip"):
            ts.append(VFlip(sec["vflip"]))
        if "color_jitter" in sec:
            ts.append(ColorJitter(**sec["color_jitter"]))
        if "gaussian_blur" in sec:
            ts.append(GaussianBlur(**sec["gaussian_blur"]))
    if sec.get("max_side") and not (train and crop_size):
        ts.append(ResizeLongest(int(sec["max_side"])))  # train+crop: đã resize ở đầu pipeline
    if pad_multiple:
        ts.append(PadToMultiple(pad_multiple))
    return Compose(ts, seed=seed)
