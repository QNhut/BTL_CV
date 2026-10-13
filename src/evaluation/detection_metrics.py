"""mAP cho bài toán một lớp (nên mAP = AP của lớp "quả"), AP kiểu all-point (VOC/COCO-continuous)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..inference.postprocess import box_iou_matrix

DEFAULT_IOU_THRS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Diện tích dưới đường precision-recall đã làm đơn điệu (all-point interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match(
    pred_boxes: Sequence[np.ndarray],
    pred_scores: Sequence[np.ndarray],
    gt_boxes: Sequence[np.ndarray],
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Gán box dự đoán với gt (greedy theo điểm). Trả về (scores, is_tp, tổng số gt)."""
    scores_all, tp_all = [], []
    n_gt = 0
    for pb, ps, gb in zip(pred_boxes, pred_scores, gt_boxes):
        pb = np.asarray(pb, np.float32).reshape(-1, 4)
        ps = np.asarray(ps, np.float32).reshape(-1)
        gb = np.asarray(gb, np.float32).reshape(-1, 4)
        n_gt += len(gb)
        if len(pb) == 0:
            continue
        order = np.argsort(-ps, kind="stable")
        pb, ps = pb[order], ps[order]
        ious = box_iou_matrix(pb, gb)
        taken = np.zeros(len(gb), bool)
        tp = np.zeros(len(pb), bool)
        for i in range(len(pb)):
            if len(gb) == 0:
                break
            cand = np.where(taken, -1.0, ious[i])
            j = int(np.argmax(cand))
            if cand[j] >= iou_thr:
                taken[j] = True
                tp[i] = True
        scores_all.append(ps)
        tp_all.append(tp)
    if not scores_all:
        return np.zeros(0, np.float32), np.zeros(0, bool), n_gt
    return np.concatenate(scores_all), np.concatenate(tp_all), n_gt


def ap_at_iou(pred_boxes, pred_scores, gt_boxes, iou_thr: float = 0.5) -> dict[str, np.ndarray | float]:
    scores, tp, n_gt = _match(pred_boxes, pred_scores, gt_boxes, iou_thr)
    if n_gt == 0:
        return {"ap": float("nan"), "precision": np.zeros(0), "recall": np.zeros(0), "scores": scores}
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(np.float64)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    return {
        "ap": average_precision(recall, precision) if len(tp) else 0.0,
        "precision": precision,
        "recall": recall,
        "scores": scores[order],
    }


def evaluate_detection(
    pred_boxes: Sequence[np.ndarray],
    pred_scores: Sequence[np.ndarray],
    gt_boxes: Sequence[np.ndarray],
    iou_thrs: Sequence[float] = DEFAULT_IOU_THRS,
) -> dict[str, float]:
    """mAP@0.5, mAP@0.75, mAP@[0.5:0.95] và precision/recall/F1 tốt nhất tại IoU 0.5."""
    aps = {}
    for t in iou_thrs:
        aps[float(t)] = ap_at_iou(pred_boxes, pred_scores, gt_boxes, float(t))["ap"]
    r50 = ap_at_iou(pred_boxes, pred_scores, gt_boxes, 0.5)
    valid = [v for v in aps.values() if not np.isnan(v)]
    out = {
        "map50": float(aps.get(0.5, r50["ap"])),
        "map75": float(aps.get(0.75, float("nan"))),
        "map": float(np.mean(valid)) if valid else float("nan"),
    }
    p, r = r50["precision"], r50["recall"]
    if len(p):
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
        k = int(np.argmax(f1))
        out.update(
            best_f1=float(f1[k]),
            precision_at_best_f1=float(p[k]),
            recall_at_best_f1=float(r[k]),
            best_f1_score_thr=float(r50["scores"][k]),
        )
    else:
        out.update(best_f1=0.0, precision_at_best_f1=0.0, recall_at_best_f1=0.0, best_f1_score_thr=float("nan"))
    return out
