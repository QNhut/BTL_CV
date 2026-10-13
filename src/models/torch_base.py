"""Lớp nền cho các model PyTorch do project tự huấn luyện (Faster R-CNN, Mask R-CNN, CSRNet)."""
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseCounter


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def array_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
    return t.to(device)


class TorchCounter(BaseCounter):
    dataset_kind: str = "detection"   # detection | segmentation | density
    train_cfg_key: str = "train"

    def __init__(self, cfg: dict, device: Optional[str] = None):
        super().__init__(cfg)
        self.device = resolve_device(device or (cfg.get("model", {}) or {}).get("device"))
        self.network: nn.Module = self._build_network().to(self.device)

    # ----- cần cài ở lớp con -----
    @abstractmethod
    def _build_network(self) -> nn.Module: ...

    @abstractmethod
    def training_step(self, batch) -> dict[str, torch.Tensor]:
        """Trả về dict có khoá 'loss' (tensor có grad) và các thành phần loss để log."""

    # ----- tiện ích chung -----
    @property
    def train_cfg(self) -> dict:
        return dict(self.cfg.get(self.train_cfg_key, {}) or {})

    @property
    def density_cfg(self) -> Optional[dict]:
        return None

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters())

    def trainable_parameters(self):
        return [p for p in self.network.parameters() if p.requires_grad]

    def train_mode(self) -> None:
        self.network.train()

    def eval_mode(self) -> None:
        self.network.eval()

    def state_dict(self) -> dict:
        return self.network.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.network.load_state_dict(state)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), "cfg_name": self.cfg.get("name")}, path)

    def load(self, weights: str) -> None:
        ckpt = torch.load(weights, map_location=self.device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.load_state_dict(state)
        self.network.to(self.device).eval()

    @torch.no_grad()
    def predict_images(self, images):  # type: ignore[override]
        was_training = self.network.training
        self.network.eval()
        try:
            return super().predict_images(images)
        finally:
            if was_training:
                self.network.train()
