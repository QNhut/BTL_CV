import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.advanced_augmentation import (
    AdaptiveCLAHE,
    CanopyShadow,
    ClusterCopyPaste,
    FarmWeatherSim,
    FoliageOcclusion,
    HighResFruitTileCrop,
    MicroFruitDownscaleSim,
    SunFlare,
    create_mosaic4,
)
from src.data.augmentation import Sample


def make_dummy_sample():
    h, w = 400, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[50:150, 50:150] = (200, 50, 50)     # Quả 1: đỏ
    img[200:300, 200:300] = (50, 180, 50)   # Quả 2: xanh
    img[120:220, 80:180] = (220, 100, 40)   # Quả 3: cam (chồng lấn)

    mask = np.zeros((h, w), dtype=np.int32)
    mask[50:150, 50:150] = 1
    mask[200:300, 200:300] = 2
    mask[120:220, 80:180] = 3

    boxes = np.array([
        [50, 50, 150, 150],
        [200, 200, 300, 300],
        [80, 120, 180, 220],
    ], dtype=np.float32)
    ids = np.array([1, 2, 3], dtype=np.int64)

    return Sample(img, boxes, ids, mask)


def test_foliage_occlusion(dummy_sample=None):
    dummy_sample = dummy_sample or make_dummy_sample()
    rng = np.random.default_rng(42)
    t = FoliageOcclusion(p=1.0)
    out = t(dummy_sample, rng)
    assert out.image.shape == dummy_sample.image.shape
    assert out.image.dtype == np.uint8
    assert len(out.boxes) >= 0


def test_cluster_copy_paste(dummy_sample=None):
    dummy_sample = dummy_sample or make_dummy_sample()
    rng = np.random.default_rng(42)
    t = ClusterCopyPaste(p=1.0)
    out = t(dummy_sample, rng)
    assert out.image.shape == dummy_sample.image.shape
    assert out.inst_mask is not None
    assert len(out.boxes) >= len(dummy_sample.boxes)


def test_canopy_shadow(dummy_sample=None):
    dummy_sample = dummy_sample or make_dummy_sample()
    rng = np.random.default_rng(42)
    t = CanopyShadow(p=1.0)
    out = t(dummy_sample, rng)
    assert out.image.shape == dummy_sample.image.shape
    assert out.image.dtype == np.uint8
    assert out.image.mean() <= dummy_sample.image.mean() + 10


def test_sun_flare(dummy_sample=None):
    dummy_sample = dummy_sample or make_dummy_sample()
    rng = np.random.default_rng(42)
    t = SunFlare(p=1.0)
    out = t(dummy_sample, rng)
    assert out.image.shape == dummy_sample.image.shape
    assert out.image.dtype == np.uint8


def test_clahe(dummy_sample=None):
    dummy_sample = dummy_sample or make_dummy_sample()
    rng = np.random.default_rng(42)
    t = AdaptiveCLAHE(p=1.0)
    out = t(dummy_sample, rng)
    assert out.image.shape == dummy_sample.image.shape
    assert out.image.dtype == np.uint8


def test_weather_sim(dummy_sample=None):
    dummy_sample = dummy_sample or make_dummy_sample()
    rng = np.random.default_rng(42)
    t = FarmWeatherSim(p=1.0)
    out = t(dummy_sample, rng)
    assert out.image.shape == dummy_sample.image.shape
    assert out.image.dtype == np.uint8


def test_high_res_tile(dummy_sample=None):
    dummy_sample = dummy_sample or make_dummy_sample()
    rng = np.random.default_rng(42)
    t = HighResFruitTileCrop(tile_size=200, min_fruits=1)
    out = t(dummy_sample, rng)
    assert out.image.shape[0] <= 400
    assert out.image.shape[1] <= 400
    assert len(out.boxes) >= 1


def test_downscale_sim(dummy_sample=None):
    dummy_sample = dummy_sample or make_dummy_sample()
    rng = np.random.default_rng(42)
    t = MicroFruitDownscaleSim(p=1.0)
    out = t(dummy_sample, rng)
    assert out.image.shape == dummy_sample.image.shape


def test_mosaic4(dummy_sample=None):
    dummy_sample = dummy_sample or make_dummy_sample()
    rng = np.random.default_rng(42)
    samples = [dummy_sample] * 4
    out = create_mosaic4(samples, target_size=(512, 512), rng=rng)
    assert out.image.shape == (512, 512, 3)
    assert out.inst_mask.shape == (512, 512)
    assert len(out.boxes) > 0
    assert np.all(out.boxes[:, 0] >= 0) and np.all(out.boxes[:, 2] <= 512)
    assert np.all(out.boxes[:, 1] >= 0) and np.all(out.boxes[:, 3] <= 512)


if __name__ == "__main__":
    print("Testing advanced augmentation transforms...")
    test_foliage_occlusion()
    test_cluster_copy_paste()
    test_canopy_shadow()
    test_sun_flare()
    test_clahe()
    test_weather_sim()
    test_high_res_tile()
    test_downscale_sim()
    test_mosaic4()
    print("ALL 9 ADVANCED AUGMENTATION TESTS PASSED SUCCESSFULLY!")

