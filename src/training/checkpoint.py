"""Lưu / nạp checkpoint an toàn (ghi tạm rồi đổi tên) để resume được sau khi Kaggle/Colab ngắt phiên."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


class CheckpointManager:
    """Quản lý `last.pt` (đầy đủ trạng thái để resume) và `best.pt` (chỉ trọng số, dùng suy luận)."""

    def __init__(self, out_dir: str | Path):
        self.dir = Path(out_dir) / "weights"
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.dir / "best.pt"

    def has_last(self) -> bool:
        return self.last_path.exists()

    def save_last(self, state: dict[str, Any]) -> None:
        save_checkpoint(self.last_path, state)

    def save_best(self, model_state: dict[str, Any], extra: Optional[dict] = None) -> None:
        save_checkpoint(self.best_path, {"model": model_state, **(extra or {})})

    def load_last(self, map_location: Any = "cpu") -> dict[str, Any]:
        return load_checkpoint(self.last_path, map_location)
