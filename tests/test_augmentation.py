import numpy as np
import pytest

from src.data import Sample, build_augmenter, masks_to_boxes
from src.data.augmentation import (
    ColorJitter,
    Compose,
    HFlip,
    PadToMultiple,
    RandomCropFixed,
    RandomScaleCrop,
    ResizeLongest,
    VFlip,
    crop_sample,
)
from src.utils import load_image, load_yaml
from src.data.annotations import read_instance_mask


@pytest.fixture()
def sample(records):
    r = records[0]
    return Sample(load_image(r.image_path), r.boxes.copy(), r.instance_ids.copy(), read_instance_mask(r.mask_path))


def _assert_consistent(s: Sample, tol: float = 1.5):
    """Box phải khớp bbox của instance mask tương ứng, và nằm trong ảnh."""
    h, w = s.image.shape[:2]
    assert s.inst_mask.shape == (h, w)
    assert len(s.boxes) == len(s.ids)
    if len(s.boxes):
        assert s.boxes.min() >= -1e-3
        assert s.boxes[:, [0, 2]].max() <= w + 1e-3 and s.boxes[:, [1, 3]].max() <= h + 1e-3
        assert ((s.boxes[:, 2] - s.boxes[:, 0]) > 0).all()
    for box, i in zip(s.boxes, s.ids):
        ys, xs = np.nonzero(s.inst_mask == i)
        if len(xs):
            ref = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            assert np.abs(ref - box).max() <= tol


def test_hflip_and_vflip_keep_consistency(sample):
    rng = np.random.default_rng(0)
    _assert_consistent(sample, tol=0.01)
    f = HFlip(1.0)(sample, rng)
    assert np.array_equal(f.image, sample.image[:, ::-1])
    _assert_consistent(f, tol=0.01)
    v = VFlip(1.0)(sample, rng)
    _assert_consistent(v, tol=0.01)
    back = HFlip(1.0)(f, rng)  # lật hai lần = ban đầu
    assert np.allclose(back.boxes, sample.boxes)


def test_flip_probability_zero_is_identity(sample):
    out = HFlip(0.0)(sample, np.random.default_rng(0))
    assert out is sample


def test_crop_sample_drops_barely_visible_boxes():
    img = np.zeros((100, 100, 3), np.uint8)
    boxes = np.array([[10, 10, 30, 30], [45, 45, 65, 65]], np.float32)
    s = Sample(img, boxes, np.array([1, 2]))
    out = crop_sample(s, 0, 0, 50, 50, min_visibility=0.3)
    # box thứ hai chỉ còn 5x5 / 20x20 = 6% diện tích -> bị loại
    assert out.boxes.tolist() == [[10, 10, 30, 30]] and out.ids.tolist() == [1]
    out2 = crop_sample(s, 0, 0, 50, 50, min_visibility=0.0)
    assert len(out2.boxes) == 2


def test_random_scale_crop_many_seeds(sample):
    for seed in range(30):
        out = RandomScaleCrop(p=1.0)(sample, np.random.default_rng(seed))
        _assert_consistent(out, tol=0.01)
        assert out.image.shape[0] <= sample.image.shape[0] and out.image.shape[1] <= sample.image.shape[1]


def test_random_crop_fixed_size_and_padding(sample):
    out = RandomCropFixed(96)(sample, np.random.default_rng(1))
    assert out.image.shape[:2] == (96, 96)
    _assert_consistent(out, tol=0.01)
    small = Sample(np.zeros((40, 50, 3), np.uint8), np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
    out2 = RandomCropFixed(64)(small, np.random.default_rng(1))
    assert out2.image.shape[:2] == (64, 64)  # ảnh nhỏ hơn được pad


def test_resize_longest_scales_boxes_and_mask(sample):
    out = ResizeLongest(128)(sample)
    assert max(out.image.shape[:2]) == 128
    _assert_consistent(out, tol=2.0)
    assert ResizeLongest(1000)(sample) is sample  # chỉ thu nhỏ, không phóng to
    up = ResizeLongest(512, only_shrink=False)(sample)
    assert max(up.image.shape[:2]) == 512
    _assert_consistent(up, tol=3.0)


def test_pad_to_multiple(sample):
    s = ResizeLongest(101)(sample)
    p = PadToMultiple(8)(s)
    assert p.image.shape[0] % 8 == 0 and p.image.shape[1] % 8 == 0
    assert p.inst_mask.shape == p.image.shape[:2]


def test_color_jitter_changes_pixels_not_geometry(sample):
    out = ColorJitter(p=1.0)(sample, np.random.default_rng(0))
    assert out.image.dtype == np.uint8 and out.image.shape == sample.image.shape
    assert not np.array_equal(out.image, sample.image)
    assert np.array_equal(out.boxes, sample.boxes)


def test_compose_reproducible_with_seed(sample):
    cfg = load_yaml("configs/augmentation.yaml")
    a = build_augmenter(cfg, train=True, seed=7)(sample)
    b = build_augmenter(cfg, train=True, seed=7)(sample)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.boxes, b.boxes)
    _assert_consistent(a, tol=2.0)


def test_build_augmenter_val_is_deterministic(sample):
    cfg = {"val": {"max_side": 100}}
    aug = build_augmenter(cfg, train=False)
    a, b = aug(sample), aug(sample)
    assert np.array_equal(a.image, b.image) and max(a.image.shape[:2]) == 100


def test_build_augmenter_density_pipeline(sample):
    cfg = load_yaml("configs/augmentation.yaml")
    aug = build_augmenter(cfg, train=True, seed=0, crop_size=64, pad_multiple=8)
    for _ in range(5):
        out = aug(sample)
        assert out.image.shape[:2] == (64, 64)
