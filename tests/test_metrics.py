import math

import numpy as np
import pytest

from src.evaluation import (
    accuracy_within,
    average_precision,
    counting_report,
    error_by_count_bin,
    evaluate_detection,
    mae,
    mape,
    r2_score,
    rmse,
    total_relative_error,
    tune_count_threshold,
    worst_cases,
    per_image_table,
)
from src.models.base import CountResult


# ---------- chỉ số đếm ----------
def test_mae_rmse_known_values():
    pred, gt = [10, 20, 30], [12, 18, 30]
    assert mae(pred, gt) == pytest.approx(4 / 3)
    assert rmse(pred, gt) == pytest.approx(math.sqrt(8 / 3))


def test_mape_ignores_zero_gt():
    assert mape([5, 3], [10, 0]) == pytest.approx(50.0)
    assert math.isnan(mape([1], [0]))


def test_total_relative_error_and_r2():
    assert total_relative_error([90, 110], [100, 100]) == pytest.approx(0.0)
    assert total_relative_error([50], [100]) == pytest.approx(-0.5)
    assert r2_score([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert math.isnan(r2_score([1, 1], [2, 2]))


def test_accuracy_within_rounds_predictions():
    assert accuracy_within([4.6, 7.2, 10.0], [5, 7, 12], tol=0) == pytest.approx(2 / 3)
    assert accuracy_within([4.6, 7.2, 10.0], [5, 7, 12], tol=2) == pytest.approx(1.0)


def test_counting_report_keys_and_empty():
    r = counting_report([1, 2], [1, 3])
    assert r["n_images"] == 2 and r["mae"] == pytest.approx(0.5) and r["bias"] == pytest.approx(-0.5)
    assert r["gt_total"] == 4 and r["pred_total"] == 3
    assert math.isnan(mae([], []))


# ---------- mAP ----------
def test_average_precision_known_curve():
    # TP rồi FP: recall [0.5, 0.5], precision [1, 0.5] -> AP = 0.5
    assert average_precision(np.array([0.5, 0.5]), np.array([1.0, 0.5])) == pytest.approx(0.5)


def test_perfect_detection_gives_map_one():
    gt = [np.array([[0, 0, 10, 10], [20, 20, 40, 40]], np.float32)]
    out = evaluate_detection(gt, [np.array([0.9, 0.8])], gt)
    assert out["map50"] == pytest.approx(1.0) and out["map"] == pytest.approx(1.0)
    assert out["best_f1"] == pytest.approx(1.0)


def test_no_predictions_gives_zero_and_no_gt_is_nan():
    gt = [np.array([[0, 0, 10, 10]], np.float32)]
    out = evaluate_detection([np.zeros((0, 4))], [np.zeros(0)], gt)
    assert out["map50"] == 0.0 and out["best_f1"] == 0.0
    nan = evaluate_detection([np.zeros((0, 4))], [np.zeros(0)], [np.zeros((0, 4))])
    assert math.isnan(nan["map50"])


def test_false_positive_with_higher_score_lowers_ap():
    gt = [np.array([[0, 0, 10, 10], [50, 50, 60, 60]], np.float32)]
    pb = [np.array([[0, 0, 10, 10], [100, 100, 110, 110], [50, 50, 60, 60]], np.float32)]
    # thứ tự theo điểm: TP(.9), FP(.8), TP(.7) -> precision [1, .5, .667], recall [.5, .5, 1]
    out = evaluate_detection(pb, [np.array([0.9, 0.8, 0.7])], gt)
    assert out["map50"] == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))


def test_duplicate_detection_counts_as_false_positive():
    gt = [np.array([[0, 0, 10, 10]], np.float32)]
    pb = [np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)]
    out = evaluate_detection(pb, [np.array([0.9, 0.8])], gt)
    assert out["map50"] == pytest.approx(1.0)  # TP trước, FP sau không làm giảm AP all-point
    assert out["precision_at_best_f1"] == pytest.approx(1.0)


def test_map_drops_with_stricter_iou():
    gt = [np.array([[0, 0, 10, 10]], np.float32)]
    shifted = [np.array([[2, 0, 12, 10]], np.float32)]  # IoU = 8/12 ≈ 0.667
    out = evaluate_detection(shifted, [np.array([0.9])], gt)
    assert out["map50"] == pytest.approx(1.0) and out["map75"] == pytest.approx(0.0)
    assert 0 < out["map"] < 1


# ---------- ngưỡng đếm & phân tích lỗi ----------
def _det(scores):
    return CountResult(boxes=np.zeros((len(scores), 4), np.float32), scores=np.array(scores, np.float32))


def test_countresult_count_at_threshold():
    r = _det([0.9, 0.6, 0.3, 0.1])
    assert r.count_at(0.5) == 2 and r.count_at(0.05) == 4 and r.count_at(0.95) == 0
    d = CountResult(density=np.ones((2, 2), np.float32), density_count=4.0)
    assert d.count_at(0.99) == 4.0  # density bỏ qua ngưỡng


def test_tune_count_threshold_picks_best():
    results = [_det([0.9, 0.8, 0.4, 0.3]), _det([0.95, 0.35, 0.2])]
    thr, best = tune_count_threshold(results, [2, 1], grid=[0.1, 0.5, 0.85])
    # thr 0.1 -> (4, 3), MAE 2; thr 0.5 -> (2, 1), MAE 0; thr 0.85 -> (1, 1), MAE 0.5
    assert thr == 0.5 and best == pytest.approx(0.0)


def test_error_by_count_bin_groups_images():
    gt = [3, 8, 30, 60]
    pred = [4, 8, 25, 70]
    rows = {r["gt_count_bin"]: r for r in error_by_count_bin(pred, gt)}
    assert rows["0-9"]["n_images"] == 2 and rows["0-9"]["mae"] == pytest.approx(0.5)
    assert rows["25-49"]["bias"] == pytest.approx(-5.0)
    assert rows["50-99"]["mae"] == pytest.approx(10.0)


def test_per_image_table_and_worst_cases():
    rows = per_image_table(["a", "b", "c"], [10, 5, 0], [10, 8, 0])
    assert rows[1]["error"] == -3 and rows[1]["rel_error_pct"] == pytest.approx(37.5)
    assert math.isnan(rows[2]["rel_error_pct"])
    assert worst_cases(rows, 1)[0]["image_id"] == "b"
