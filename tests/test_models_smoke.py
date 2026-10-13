"""Smoke test các model PyTorch trên CPU với ảnh nhỏ, không tải trọng số pretrained."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.data import build_augmenter  # noqa: E402
from src.data.dataset import build_dataset, get_collate_fn  # noqa: E402
from src.models import build_model, get_model_class  # noqa: E402
from src.models.base import CountResult  # noqa: E402
from src.models.csrnet import CSRNetModule  # noqa: E402
from src.utils import load_image, load_yaml  # noqa: E402

SMALL = {"min_size": 128, "max_size": 256, "pretrained": False, "device": "cpu"}


def _batch(records, kind, n=2, aug=None):
    ds = build_dataset(kind, records[:n], aug, {"downsample": 8})
    return get_collate_fn(kind)([ds[i] for i in range(n)])


def _cfg(t, **extra):
    base = {
        "faster_rcnn": {"name": "frcnn", "type": "faster_rcnn", "family": "detection",
                        "model": {**SMALL, "box_detections_per_img": 50},
                        "inference": {"min_score": 0.0, "count_threshold": 0.0}},
        "mask_rcnn": {"name": "mrcnn", "type": "mask_rcnn", "family": "segmentation",
                      "model": {**SMALL, "box_detections_per_img": 20},
                      "inference": {"min_score": 0.0, "count_threshold": 0.0}},
        "csrnet": {"name": "csr", "type": "csrnet", "family": "density",
                   "model": {"pretrained": False, "density_scale": 100.0, "max_side": 128, "device": "cpu"},
                   "train": {"count_loss_weight": 0.1}, "inference": {}},
    }[t]
    base.update(extra)
    return base


def test_csrnet_module_stride_8():
    net = CSRNetModule(pretrained=False).eval()
    out = net(torch.rand(1, 3, 96, 160))
    assert out.shape == (1, 1, 12, 20)


def test_csrnet_training_step_and_backward(records):
    m = build_model(_cfg("csrnet"))
    aug = build_augmenter(load_yaml("configs/augmentation.yaml"), train=True, seed=0, crop_size=64, pad_multiple=8)
    out = m.training_step(_batch(records, "density", 2, aug))
    assert torch.isfinite(out["loss"]) and {"mse", "count_l1"} <= set(out)
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.network.parameters())


def test_csrnet_predict_returns_density_result(records):
    m = build_model(_cfg("csrnet"))
    img = load_image(records[0].image_path)
    res = m.predict_images([img, img])
    assert len(res) == 2 and all(isinstance(r, CountResult) and r.is_density for r in res)
    assert res[0].density.ndim == 2 and res[0].count >= 0
    assert res[0].density.sum() == pytest.approx(res[0].count, rel=1e-4)


def test_faster_rcnn_train_and_predict(records):
    m = build_model(_cfg("faster_rcnn"))
    m.train_mode()
    out = m.training_step(_batch(records, "detection", 2))
    assert torch.isfinite(out["loss"]) and "loss_classifier" in out
    out["loss"].backward()
    m.eval_mode()
    res = m.predict_images([load_image(records[0].image_path)])[0]
    assert res.boxes.shape[1] == 4 and len(res.boxes) == len(res.scores)
    assert res.boxes.dtype == np.float32


def test_mask_rcnn_train_and_predict(records):
    m = build_model(_cfg("mask_rcnn"))
    m.train_mode()
    out = m.training_step(_batch(records, "segmentation", 2))
    assert torch.isfinite(out["loss"]) and "loss_mask" in out
    m.eval_mode()
    res = m.predict_images([load_image(records[0].image_path)])[0]
    assert res.masks is not None and res.masks.dtype == bool
    assert len(res.masks) == int(res.count_at())  # chỉ giữ mask cho quả vượt ngưỡng đếm


def test_predict_images_restores_training_mode(records):
    m = build_model(_cfg("csrnet"))
    m.train_mode()
    m.predict_images([load_image(records[0].image_path)])
    assert m.network.training


def test_save_and_load_roundtrip(records, tmp_path):
    a = build_model(_cfg("csrnet"))
    a.save(tmp_path / "w.pt")
    b = build_model(_cfg("csrnet"))
    b.load(str(tmp_path / "w.pt"))
    img = load_image(records[0].image_path)
    assert a.predict_image(img).count == pytest.approx(b.predict_image(img).count)


def test_registry_errors():
    with pytest.raises(KeyError):
        get_model_class("nonexistent")
