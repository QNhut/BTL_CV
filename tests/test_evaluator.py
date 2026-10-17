"""Test evaluator đầu-cuối với model giả (oracle) để không cần tải trọng số."""
import numpy as np
import pytest

from src.evaluation import evaluate_model, summarize
from src.models.base import BaseCounter, CountResult


class OracleDetector(BaseCounter):
    """Trả đúng box GT (hoặc GT + nhiễu) theo thứ tự ảnh được gọi."""

    family = "detection"

    def __init__(self, records, extra_fp=0, drop=0):
        super().__init__({"name": "oracle", "inference": {"count_threshold": 0.5, "min_score": 0.0}})
        self.queue = list(records)
        self.all = list(records)
        self.extra_fp, self.drop = extra_fp, drop

    def _predict_raw(self, images):
        out = []
        for _ in images:
            if not self.queue:
                self.queue = list(self.all)
            r = self.queue.pop(0)
            boxes = r.boxes[self.drop:]
            scores = np.full(len(boxes), 0.9, np.float32)
            if self.extra_fp:
                fp = np.tile(np.array([[0, 0, 3, 3]], np.float32), (self.extra_fp, 1))
                boxes = np.concatenate([boxes, fp])
                scores = np.concatenate([scores, np.full(self.extra_fp, 0.1, np.float32)])
            out.append(CountResult(boxes=boxes, scores=scores, score_thr=self.count_threshold))
        return out

    def load(self, weights):
        pass


class OracleDensity(BaseCounter):
    family = "density"

    def __init__(self, records, bias=0.0):
        super().__init__({"name": "oracle_density"})
        self.queue, self.bias = list(records), bias

    def _predict_raw(self, images):
        out = []
        for _ in images:
            r = self.queue.pop(0)
            d = np.zeros((4, 4), np.float32)
            d[0, 0] = r.count + self.bias
            out.append(CountResult(density=d, density_count=float(d.sum())))
        return out

    def load(self, weights):
        pass


def test_perfect_detector_scores_perfectly(records):
    rep = evaluate_model(OracleDetector(records), records, batch_size=3)
    assert rep["counting"]["mae"] == 0 and rep["counting"]["rmse"] == 0
    assert rep["detection"]["map50"] == pytest.approx(1.0)
    assert len(rep["per_image"]) == len(records)
    assert "oracle" in summarize(rep)


def test_low_score_false_positives_hurt_map_but_not_counting_at_threshold(records):
    rep = evaluate_model(OracleDetector(records, extra_fp=2), records, batch_size=4)
    assert rep["counting"]["mae"] == 0          # FP điểm 0.1 < ngưỡng đếm 0.5
    assert rep["detection"]["map50"] == pytest.approx(1.0)  # FP xếp sau toàn bộ TP nên AP vẫn 1


def test_missing_detections_increase_mae(records):
    rep = evaluate_model(OracleDetector(records, drop=1), records, batch_size=4)
    # mỗi ảnh có >= 1 quả nên thiếu đúng 1 quả/ảnh
    assert rep["counting"]["mae"] == pytest.approx(1.0)
    assert rep["counting"]["bias"] == pytest.approx(-1.0)
    assert rep["detection"]["map50"] < 1.0


def test_threshold_tuning_on_val_split_is_applied(records):
    # FP có điểm 0.1; đặt ngưỡng cố định thấp 0.05 làm MAE kém, tuning phải chọn lại ngưỡng cao hơn
    model = OracleDetector(records, extra_fp=2)
    model.set_count_threshold(0.05)
    rep = evaluate_model(model, records, tune_on=records, batch_size=4)
    assert rep["threshold_tuning"]["best_threshold"] > 0.1
    assert rep["counting"]["mae"] == 0
    assert model.count_threshold == rep["count_threshold"]


def test_density_model_has_no_detection_metrics(records):
    rep = evaluate_model(OracleDensity(records, bias=2.0), records, batch_size=5)
    assert "detection" not in rep and "count_threshold" not in rep
    assert rep["counting"]["mae"] == pytest.approx(2.0)
    assert rep["error_by_count_bin"]


def test_return_results_and_speed(records):
    m = OracleDetector(records[:4])
    rep, results = evaluate_model(m, records[:4], with_speed=True, speed_images=2, return_results=True)
    assert len(results) == 4 and rep["speed"]["fps"] > 0


def test_metric_inference_does_not_retain_full_resolution_masks(records):
    from src.evaluation.evaluator import run_inference

    class MaskOracle(OracleDetector):
        def _predict_raw(self, images):
            results = super()._predict_raw(images)
            for result, image in zip(results, images):
                result.masks = np.zeros((1, *image.shape[:2]), dtype=bool)
            return results

    model = MaskOracle(records[:2])
    compact = run_inference(model, records[:2], batch_size=1)
    assert all(result.masks is None for result in compact)
    detailed = run_inference(model, records[:2], batch_size=1, keep_visuals=True)
    assert all(result.masks is not None for result in detailed)


def test_metric_inference_keeps_density_count_without_storing_all_maps(records):
    from src.evaluation.evaluator import run_inference
    results = run_inference(OracleDensity(records[:2]), records[:2], batch_size=1)
    assert all(result.density is None for result in results)
    assert [result.count for result in results] == [record.count for record in records[:2]]
