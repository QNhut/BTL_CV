"""Chia train/val/test theo ảnh, có thể stratify theo số quả mỗi ảnh để ba tập
có phân bố mật độ quả tương tự nhau (quan trọng khi so sánh MAE)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.io import read_json, write_json
from .annotations import ImageRecord, load_records


def make_splits(
    records: list[ImageRecord],
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    stratify_by_count: bool = True,
    bin_size: int = 10,
) -> dict[str, list[str]]:
    """Trả về {"train": [image_id...], "val": [...], "test": [...]} (rời nhau, phủ kín)."""
    if val < 0 or test < 0 or val + test >= 1:
        raise ValueError("val + test phải nằm trong [0, 1)")
    ids = [r.image_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("image_id bị trùng; cần đặt tên ảnh duy nhất")

    rng = np.random.default_rng(seed)
    n = len(records)
    if stratify_by_count:
        order = np.argsort([r.count for r in records], kind="stable")
        groups = np.array_split(order, max(1, n // bin_size))
    else:
        groups = [np.arange(n)]

    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    seen = test_done = val_done = 0
    for g in groups:
        g = rng.permutation(g)
        seen += len(g)
        # làm tròn tích lũy để tổng số ảnh mỗi split khớp đúng tỉ lệ yêu cầu
        n_test = min(len(g), max(0, int(round(test * seen)) - test_done))
        n_val = min(len(g) - n_test, max(0, int(round(val * seen)) - val_done))
        test_done += n_test
        val_done += n_val
        out["test"] += [ids[i] for i in g[:n_test]]
        out["val"] += [ids[i] for i in g[n_test : n_test + n_val]]
        out["train"] += [ids[i] for i in g[n_test + n_val :]]
    for k in out:
        out[k] = sorted(out[k])
    return out


def save_splits(splits: dict[str, list[str]], path: str | Path) -> None:
    write_json(splits, path)


def load_splits(path: str | Path) -> dict[str, list[str]]:
    return read_json(path)


def select_records(records: list[ImageRecord], image_ids: list[str]) -> list[ImageRecord]:
    by_id = {r.image_id: r for r in records}
    missing = [i for i in image_ids if i not in by_id]
    if missing:
        raise KeyError(f"Thiếu {len(missing)} image_id trong records, ví dụ: {missing[:3]}")
    return [by_id[i] for i in image_ids]


def load_split_records(processed_dir: str | Path, split: str) -> list[ImageRecord]:
    """Nạp records của một split từ thư mục đã chạy `scripts/prepare_data.py`."""
    processed_dir = Path(processed_dir)
    records = load_records(processed_dir / "records.json")
    splits = load_splits(processed_dir / "splits.json")
    if split not in splits:
        raise KeyError(f"Split '{split}' không tồn tại; có: {list(splits)}")
    return select_records(records, splits[split])
