import numpy as np
import pytest

from src.data import boxes_to_density, boxes_to_points


def test_points_are_box_centers():
    p = boxes_to_points(np.array([[0, 0, 10, 20], [4, 6, 8, 10]], np.float32))
    assert p.tolist() == [[5, 10], [6, 8]]


def test_density_sum_equals_count():
    boxes = np.array([[10, 10, 30, 30], [50, 40, 70, 60], [100, 80, 112, 96]], np.float32)
    d = boxes_to_density(boxes, (128, 160), downsample=8)
    assert d.shape == (16, 20) and d.dtype == np.float32
    assert d.sum() == pytest.approx(3.0, abs=1e-4)


def test_density_sum_preserved_at_border():
    # quả sát biên: cửa sổ Gaussian bị cắt nhưng tổng vẫn phải là 1
    boxes = np.array([[0, 0, 8, 8], [152, 120, 160, 128]], np.float32)
    d = boxes_to_density(boxes, (128, 160), downsample=8)
    assert d.sum() == pytest.approx(2.0, abs=1e-4)


def test_density_empty_and_outside():
    assert boxes_to_density(np.zeros((0, 4)), (64, 64)).sum() == 0
    outside = np.array([[500, 500, 520, 520]], np.float32)
    assert boxes_to_density(outside, (64, 64)).sum() == 0  # ngoài ảnh thì bỏ qua


def test_density_peak_near_center_and_sigma_modes():
    box = np.array([[60, 60, 84, 84]], np.float32)  # tâm (72, 72) -> ô (9, 9)
    d = boxes_to_density(box, (160, 160), downsample=8)
    py, px = np.unravel_index(d.argmax(), d.shape)
    assert (py, px) in {(8, 8), (8, 9), (9, 8), (9, 9)}
    wide = boxes_to_density(box, (160, 160), downsample=8, fixed_sigma=3.0)
    assert wide.max() < d.max() and wide.sum() == pytest.approx(1.0, abs=1e-4)


def test_density_matches_dataset_counts(records):
    for r in records[:10]:
        d = boxes_to_density(r.boxes, (r.height, r.width))
        assert d.sum() == pytest.approx(r.count, abs=1e-3)
