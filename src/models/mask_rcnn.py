"""Adapter Mask R-CNN (torchvision): tách từng quả bằng mask rồi đếm số instance.

Cần annotation dạng instance mask (ví dụ MinneApple). Mask chỉ được giữ lại cho
các quả vượt ngưỡng đếm để tránh tốn bộ nhớ.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import CountResult
from .torch_base import TorchCounter, array_to_tensor


class MaskRCNNCounter(TorchCounter):
    family = "segmentation"
    dataset_kind = "segmentation"

    def _build_network(self) -> nn.Module:
        from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        m = self.cfg.get("model", {}) or {}
        builder = maskrcnn_resnet50_fpn_v2 if m.get("variant", "v2") == "v2" else maskrcnn_resnet50_fpn
        net = builder(
            weights="DEFAULT" if m.get("pretrained", True) else None,
            weights_backbone=None,
            min_size=int(m.get("min_size", 800)),
            max_size=int(m.get("max_size", 1333)),
            box_detections_per_img=int(m.get("box_detections_per_img", 300)),
            box_score_thresh=float(m.get("box_score_thresh", 0.05)),
            box_nms_thresh=float(m.get("box_nms_thresh", 0.5)),
        )
        in_f = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
        in_m = net.roi_heads.mask_predictor.conv5_mask.in_channels
        net.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, 2)
        self.mask_threshold = float(m.get("mask_threshold", 0.5))
        return net

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        images, targets = batch
        images = [i.to(self.device) for i in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        losses = self.network(images, targets)
        total = sum(losses.values())
        return {"loss": total, **{k: v.detach() for k, v in losses.items()}}

    def _predict_raw(self, images: list[np.ndarray]) -> list[CountResult]:
        out = []
        thr = self.count_threshold
        for im in images:
            pred = self.network([array_to_tensor(im, self.device)])[0]
            scores = pred["scores"].cpu().numpy().astype(np.float32)
            boxes = pred["boxes"].cpu().numpy().astype(np.float32)
            keep = scores >= self.min_score
            sel_count = (pred["scores"] >= thr).nonzero(as_tuple=True)[0]
            masks = (pred["masks"][sel_count, 0] > self.mask_threshold).cpu().numpy()
            out.append(
                CountResult(boxes=boxes[keep], scores=scores[keep], masks=masks, score_thr=thr)
            )
        return out
