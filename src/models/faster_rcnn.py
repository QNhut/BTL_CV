"""Adapter Faster R-CNN (torchvision, ResNet-50 FPN) cho bài toán 1 lớp "quả"."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import CountResult
from .torch_base import TorchCounter, array_to_tensor


class FasterRCNNCounter(TorchCounter):
    family = "detection"
    dataset_kind = "detection"

    def _build_network(self) -> nn.Module:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        m = self.cfg.get("model", {}) or {}
        builder = fasterrcnn_resnet50_fpn_v2 if m.get("variant", "v2") == "v2" else fasterrcnn_resnet50_fpn
        net = builder(
            weights="DEFAULT" if m.get("pretrained", True) else None,
            weights_backbone=None,
            min_size=int(m.get("min_size", 800)),
            max_size=int(m.get("max_size", 1333)),
            box_detections_per_img=int(m.get("box_detections_per_img", 500)),
            box_score_thresh=float(m.get("box_score_thresh", 0.05)),
            box_nms_thresh=float(m.get("box_nms_thresh", 0.5)),
            rpn_pre_nms_top_n_test=int(m.get("rpn_pre_nms_top_n_test", 2000)),
            rpn_post_nms_top_n_test=int(m.get("rpn_post_nms_top_n_test", 1000)),
        )
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # nền + quả
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
        for im in images:  # xử lý từng ảnh để giữ bộ nhớ ổn định với ảnh lớn
            pred = self.network([array_to_tensor(im, self.device)])[0]
            boxes = pred["boxes"].cpu().numpy().astype(np.float32)
            scores = pred["scores"].cpu().numpy().astype(np.float32)
            keep = scores >= self.min_score
            out.append(CountResult(boxes=boxes[keep], scores=scores[keep], score_thr=self.count_threshold))
        return out
