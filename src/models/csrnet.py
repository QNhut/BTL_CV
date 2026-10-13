"""CSRNet (Li et al., CVPR 2018) hồi quy density map: frontend VGG16 + backend dilated conv.

Số quả dự đoán = tổng density map. Stride đầu ra = 8.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.augmentation import PadToMultiple, ResizeLongest, Sample
from .base import CountResult
from .torch_base import TorchCounter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CSRNetModule(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        # features[0:23] = conv1_1 ... conv4_3 + ReLU, gồm 3 max-pool => stride 8
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        layers, in_c = [], 512
        for out_c in (512, 512, 512, 256, 128, 64):
            layers += [nn.Conv2d(in_c, out_c, 3, padding=2, dilation=2), nn.ReLU(inplace=True)]
            in_c = out_c
        self.backend = nn.Sequential(*layers)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) trong [0,1] -> density (B,1,H/8,W/8)."""
        x = (x - self.mean) / self.std
        return self.output_layer(self.backend(self.frontend(x)))


class CSRNetCounter(TorchCounter):
    family = "density"
    dataset_kind = "density"
    supports_tiling = False

    def _build_network(self) -> nn.Module:
        m = self.cfg.get("model", {}) or {}
        self.density_scale = float(m.get("density_scale", 100.0))
        self.max_side = int(m.get("max_side", 1024))
        return CSRNetModule(pretrained=bool(m.get("pretrained", True)))

    @property
    def density_cfg(self) -> dict:
        return dict(self.cfg.get("density", {}) or {})

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        imgs, dens, counts = batch
        imgs, dens, counts = imgs.to(self.device), dens.to(self.device), counts.to(self.device)
        pred = self.network(imgs)
        if pred.shape[-2:] != dens.shape[-2:]:  # phòng lệch 1 ô do làm tròn
            h, w = min(pred.shape[-2], dens.shape[-2]), min(pred.shape[-1], dens.shape[-1])
            pred, dens = pred[..., :h, :w], dens[..., :h, :w]
        mse = F.mse_loss(pred, dens * self.density_scale)
        pred_count = F.relu(pred).sum(dim=(1, 2, 3)) / self.density_scale
        count_l1 = F.l1_loss(pred_count, counts)
        w = float(self.train_cfg.get("count_loss_weight", 0.1))
        return {"loss": mse + w * count_l1, "mse": mse.detach(), "count_l1": count_l1.detach()}

    def _predict_raw(self, images: list[np.ndarray]) -> list[CountResult]:
        out = []
        empty = np.zeros((0, 4), np.float32)
        for im in images:
            s = Sample(im, empty, np.zeros((0,), np.int64))
            s = PadToMultiple(8)(ResizeLongest(self.max_side)(s))
            x = torch.from_numpy(np.ascontiguousarray(s.image)).permute(2, 0, 1).float().div_(255.0)
            pred = F.relu(self.network(x[None].to(self.device)))[0, 0] / self.density_scale
            dens = pred.cpu().numpy().astype(np.float32)
            out.append(CountResult(density=dens, density_count=float(dens.sum())))
        return out
