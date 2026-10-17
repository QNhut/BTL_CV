"""Chạy Trainer thật 2 epoch trên CSRNet (CPU, ảnh nhỏ) để kiểm tra vòng huấn luyện, checkpoint, resume."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models import build_model  # noqa: E402
from src.training import CheckpointManager, Trainer  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def _cfg(epochs):
    return {
        "name": "csr_test", "type": "csrnet", "family": "density",
        "model": {"pretrained": False, "density_scale": 100.0, "max_side": 128, "device": "cpu"},
        "density": {"downsample": 8, "sigma_scale": 0.25, "min_sigma": 1.0},
        "inference": {},
        "train": {"epochs": epochs, "batch_size": 2, "num_workers": 0, "crop_size": 64, "optimizer": "adamw",
                  "lr": 1e-4, "scheduler": "cosine", "warmup_epochs": 1, "grad_clip": 5.0, "amp": False,
                  "val_every": 1, "patience": 0, "seed": 0, "count_loss_weight": 0.1},
    }


def test_trainer_runs_saves_and_resumes(records, tmp_path):
    aug = load_yaml("configs/augmentation.yaml")
    aug["val"]["max_side"] = 96
    train, val = records[:6], records[6:9]

    t1 = Trainer(build_model(_cfg(2)), train, val, aug, tmp_path)
    summary = t1.fit()
    assert len(summary["history"]) == 2
    assert all("val_mae" in h and "train_loss" in h for h in summary["history"])
    assert summary["best_val_mae"] == min(h["val_mae"] for h in summary["history"])
    ck = CheckpointManager(tmp_path)
    assert ck.last_path.exists() and ck.best_path.exists()
    assert (tmp_path / "training/history.json").exists() and (tmp_path / "training/history.csv").exists()
    assert (tmp_path / 'training/loss_curves.png').exists()
    assert (tmp_path / 'training/accuracy_curves.png').exists()
    assert all(0 <= row['val_acc_exact'] <= 1 for row in summary['history'])

    # resume: mở rộng lên 3 epoch, chỉ chạy thêm epoch cuối
    t2 = Trainer(build_model(_cfg(3)), train, val, aug, tmp_path)
    s2 = t2.fit(resume=True)
    assert [h["epoch"] for h in s2["history"]] == [0, 1, 2]
    assert t2.start_epoch == 2


def test_trainer_early_stopping(records, tmp_path):
    aug = load_yaml("configs/augmentation.yaml")
    cfg = _cfg(10)
    cfg["train"]["patience"] = 1
    cfg["train"]["lr"] = 0.0  # không học -> không cải thiện -> dừng sớm
    s = Trainer(build_model(cfg), records[:4], records[4:6], aug, tmp_path).fit()
    assert len(s["history"]) < 10


def test_density_crop_must_be_multiple_of_8(records, tmp_path):
    cfg = _cfg(1)
    cfg["train"]["crop_size"] = 60
    with pytest.raises(ValueError, match="bội số của 8"):
        Trainer(build_model(cfg), records[:2], records[2:3], load_yaml("configs/augmentation.yaml"), tmp_path)
