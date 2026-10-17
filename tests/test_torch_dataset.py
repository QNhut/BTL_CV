import pytest

torch = pytest.importorskip("torch")

from src.data import build_augmenter  # noqa: E402
from src.data.dataset import (  # noqa: E402
    FruitDensityDataset,
    FruitDetectionDataset,
    build_dataset,
    density_collate,
    detection_collate,
)
from src.utils import load_yaml  # noqa: E402


def test_detection_dataset_target_format(records):
    ds = FruitDetectionDataset(records)
    img, tgt = ds[0]
    n = records[0].count
    assert img.shape == (3, records[0].height, records[0].width) and img.dtype == torch.float32
    assert 0 <= img.min() and img.max() <= 1
    assert tgt["boxes"].shape == (n, 4) and tgt["labels"].tolist() == [1] * n
    assert "masks" not in tgt


def test_segmentation_dataset_masks_match_boxes(records):
    ds = FruitDetectionDataset(records, with_masks=True)
    img, tgt = ds[1]
    n = records[1].count
    assert tgt["masks"].shape == (n, img.shape[1], img.shape[2]) and tgt["masks"].dtype == torch.uint8
    for m, b in zip(tgt["masks"], tgt["boxes"]):
        ys, xs = torch.nonzero(m, as_tuple=True)
        assert int(xs.min()) == int(b[0]) and int(ys.max()) + 1 == int(b[3])


def test_segmentation_requires_masks(records):
    from src.data import ImageRecord
    import numpy as np

    r = records[0]
    no_mask = ImageRecord(r.image_id, r.image_path, r.width, r.height, r.boxes, r.instance_ids, mask_path=None)
    with pytest.raises(ValueError, match="instance mask"):
        FruitDetectionDataset([no_mask], with_masks=True)[0]


def test_detection_dataset_with_augmentation_keeps_targets_valid(records):
    aug = build_augmenter(load_yaml("configs/augmentation.yaml"), train=True, seed=0)
    ds = FruitDetectionDataset(records, aug, with_masks=True)
    for i in range(len(records)):
        img, tgt = ds[i]
        n = len(tgt["boxes"])
        assert tgt["masks"].shape == (n, img.shape[1], img.shape[2])
        assert tgt["labels"].shape == (n,)
        if n:
            assert (tgt["boxes"][:, 2] > tgt["boxes"][:, 0]).all()


def test_density_dataset_sum_equals_count_and_stride(records):
    ds = FruitDensityDataset(records[:5])
    for i in range(5):
        img, dens, cnt = ds[i]
        assert dens.shape[0] == 1 and dens.shape[1] == -(-img.shape[1] // 8)
        assert float(dens.sum()) == pytest.approx(float(cnt), abs=1e-3)


def test_density_dataset_with_crop_has_fixed_size(records):
    aug = build_augmenter(load_yaml("configs/augmentation.yaml"), train=True, seed=0, crop_size=64, pad_multiple=8)
    ds = FruitDensityDataset(records, aug)
    for i in range(6):
        img, dens, cnt = ds[i]
        assert img.shape == (3, 64, 64) and dens.shape == (1, 8, 8)
        assert float(dens.sum()) == pytest.approx(float(cnt), abs=1e-3)


def test_collate_functions(records):
    det = detection_collate([FruitDetectionDataset(records)[i] for i in range(3)])
    assert len(det) == 2 and len(det[0]) == 3
    ds = FruitDensityDataset(records)
    batch = density_collate([ds[0], ds[1]])
    assert batch[0].shape[0] == 2 and batch[1].shape[-1] == batch[0].shape[-1] // 8 and batch[2].shape == (2,)


def test_build_dataset_factory(records):
    assert isinstance(build_dataset("detection", records), FruitDetectionDataset)
    assert build_dataset("segmentation", records).need_mask
    assert isinstance(build_dataset("density", records), FruitDensityDataset)
    with pytest.raises(ValueError):
        build_dataset("nope", records)


def test_density_collate_odd_image_dimensions():
    import numpy as np
    from src.data import ImageRecord
    from src.data.dataset import image_to_tensor
    from src.data.density import boxes_to_density

    batch = []
    for h, w in ((65, 67), (71, 73)):
        image = image_to_tensor(np.zeros((h, w, 3), np.uint8))
        density = torch.from_numpy(boxes_to_density(np.array([[1, 1, 9, 9]]), (h, w)))[None]
        batch.append((image, density, torch.tensor(1.0)))
    images, density, counts = density_collate(batch)
    assert images.shape == (2, 3, 71, 73)
    assert density.shape == (2, 1, 9, 10)
    assert torch.allclose(density.sum((1, 2, 3)), counts)
