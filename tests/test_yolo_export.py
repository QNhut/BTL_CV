import numpy as np
import pytest
import os

from src.data import boxes_to_yolo_lines, export_yolo_dataset
from src.utils import load_yaml


def test_boxes_to_yolo_lines_normalised():
    lines = boxes_to_yolo_lines(np.array([[10, 20, 30, 60]], np.float32), width=100, height=200)
    assert lines == ["0 0.200000 0.200000 0.200000 0.200000"]


def test_boxes_to_yolo_lines_clips_and_drops_degenerate():
    lines = boxes_to_yolo_lines(np.array([[-5, -5, 10, 10], [50, 50, 50, 60], [90, 90, 120, 120]], np.float32), 100, 100)
    assert len(lines) == 2  # box suy biến bị bỏ
    assert lines[0] == "0 0.050000 0.050000 0.100000 0.100000"  # [-5,-5,10,10] cắt thành [0,0,10,10]


def test_export_creates_layout_and_data_yaml(records, tmp_path):
    yaml_path = export_yolo_dataset({"train": records[:4], "val": records[4:6], "test": records[6:8]}, tmp_path / "ds")
    cfg = load_yaml(yaml_path)
    assert cfg["train"] == "images/train" and cfg["val"] == "images/val" and cfg["test"] == "images/test"
    assert cfg["names"] == {0: "fruit"}
    for split, recs in (("train", records[:4]), ("val", records[4:6])):
        for r in recs:
            assert (tmp_path / "ds/images" / split / f"{r.image_id}.png").exists()
            n_lines = len((tmp_path / "ds/labels" / split / f"{r.image_id}.txt").read_text().split("\n")) - 1
            assert n_lines == r.count


def test_export_symlink_mode(records, tmp_path):
    if os.name == "nt":
        try:
            (tmp_path / "probe").symlink_to(records[0].image_path)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows requires Developer Mode or symlink privileges")
            raise
    export_yolo_dataset({"train": records[:2], "val": records[2:3]}, tmp_path / "ds", symlink=True)
    assert (tmp_path / "ds/images/train" / f"{records[0].image_id}.png").is_symlink()


def test_reexport_removes_images_moved_to_other_split(records, tmp_path):
    destination = tmp_path / "ds"
    export_yolo_dataset({"train": records[:2], "val": records[2:3]}, destination)
    export_yolo_dataset({"train": records[1:2], "val": [records[0], records[2]]}, destination)
    assert not (destination / 'images/train' / f'{records[0].image_id}.png').exists()
    assert not (destination / 'labels/train' / f'{records[0].image_id}.txt').exists()
    assert (destination / 'images/val' / f'{records[0].image_id}.png').exists()
