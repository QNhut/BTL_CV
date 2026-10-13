"""Hậu xử lý thuần numpy: lọc điểm, NMS, chia tile (kiểu SAHI) và gộp kết quả."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU từng cặp giữa a[N,4] và b[M,4] (xyxy) -> [N,M]."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Greedy NMS; trả về chỉ số giữ lại, sắp theo điểm giảm dần."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = box_iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, np.int64)


def filter_by_score(boxes: np.ndarray, scores: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(scores) >= thr
    return np.asarray(boxes)[m], np.asarray(scores)[m]


def tile_slices(height: int, width: int, tile: int, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Danh sách (x0, y0, x1, y1) phủ kín ảnh, các tile chồng nhau `overlap`."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap phải nằm trong [0, 1)")
    stride = max(1, int(round(tile * (1 - overlap))))

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        s = list(range(0, n - tile + 1, stride))
        if s[-1] != n - tile:
            s.append(n - tile)
        return s

    return [
        (x0, y0, min(x0 + tile, width), min(y0 + tile, height))
        for y0 in starts(height)
        for x0 in starts(width)
    ]


def merge_tile_detections(
    per_tile: Sequence[tuple[np.ndarray, np.ndarray]],
    offsets: Sequence[tuple[int, int]],
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Dịch box của từng tile về hệ tọa độ ảnh gốc rồi NMS để bỏ trùng ở vùng chồng."""
    all_b, all_s = [], []
    for (b, s), (ox, oy) in zip(per_tile, offsets):
        b = np.asarray(b, np.float32).reshape(-1, 4)
        if len(b):
            b = b + np.array([ox, oy, ox, oy], np.float32)
            all_b.append(b)
            all_s.append(np.asarray(s, np.float32).reshape(-1))
    if not all_b:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    boxes, scores = np.concatenate(all_b), np.concatenate(all_s)
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep], scores[keep]
