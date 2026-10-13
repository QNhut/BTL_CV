import numpy as np
import pytest

from src.inference.postprocess import (
    box_iou_matrix,
    filter_by_score,
    merge_tile_detections,
    nms,
    tile_slices,
)
from src.models.base import BaseCounter, CountResult


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    iou = box_iou_matrix(a, b)[0]
    assert iou[0] == pytest.approx(1.0) and iou[1] == pytest.approx(1 / 3) and iou[2] == 0
    assert box_iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_nms_keeps_highest_score_of_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    keep = nms(boxes, np.array([0.6, 0.9, 0.5]), iou_thr=0.5)
    assert keep.tolist() == [1, 2]
    assert nms(np.zeros((0, 4)), np.zeros(0)).tolist() == []


def test_filter_by_score():
    b, s = filter_by_score(np.arange(8).reshape(2, 4), np.array([0.2, 0.8]), 0.5)
    assert b.shape == (1, 4) and s.tolist() == [0.8]


@pytest.mark.parametrize("h,w,tile,ov", [(100, 100, 40, 0.2), (720, 1280, 640, 0.2), (50, 300, 100, 0.5), (30, 30, 64, 0.2)])
def test_tile_slices_cover_whole_image(h, w, tile, ov):
    cover = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in tile_slices(h, w, tile, ov):
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert (x1 - x0) <= tile and (y1 - y0) <= tile
        cover[y0:y1, x0:x1] = True
    assert cover.all()


def test_tile_slices_rejects_bad_overlap():
    with pytest.raises(ValueError):
        tile_slices(100, 100, 50, 1.0)


def test_merge_tiles_removes_duplicates_in_overlap():
    # cùng một quả (toàn cục: 90-110, 10-30) được hai tile nhìn thấy
    t1 = (np.array([[90, 10, 110, 30]], np.float32), np.array([0.9], np.float32))
    t2 = (np.array([[0, 10, 20, 30]], np.float32), np.array([0.7], np.float32))  # tile bắt đầu ở x=90... lệch 90
    boxes, scores = merge_tile_detections([t1, t2], [(0, 0), (90, 0)], iou_thr=0.5)
    assert len(boxes) == 1 and scores[0] == pytest.approx(0.9)
    assert merge_tile_detections([], [])[0].shape == (0, 4)


class _FixedDetector(BaseCounter):
    """Mỗi tile luôn thấy 1 quả ở góc trên trái của tile."""

    def __init__(self):
        super().__init__({"name": "fixed", "inference": {"count_threshold": 0.5, "nms_iou": 0.5,
                                                         "tile": {"enabled": True, "size": 64, "overlap": 0.25}}})

    def _predict_raw(self, images):
        return [CountResult(boxes=np.array([[2, 2, 12, 12]], np.float32),
                            scores=np.array([0.9], np.float32), score_thr=0.5) for _ in images]

    def load(self, weights):
        pass


def test_base_counter_tiled_prediction_merges_to_global_coordinates():
    m = _FixedDetector()
    img = np.zeros((130, 200, 3), np.uint8)
    n_tiles = len(tile_slices(130, 200, 64, 0.25))
    res = m.predict_image(img)
    assert res.count == n_tiles  # các quả ở vị trí tile khác nhau nên không bị NMS gộp
    assert res.boxes[:, 0].max() > 100  # box đã được dịch về tọa độ ảnh gốc
    m.inference_cfg["tile"]["enabled"] = False
    assert m.predict_image(img).count == 1
