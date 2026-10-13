"""Chạy thật scripts/prepare_data.py trên dataset giả (không cần torch)."""
import subprocess
import sys
from pathlib import Path

from src.data import load_split_records
from src.utils import load_yaml, read_json, save_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_data_end_to_end(synthetic, tmp_path):
    images_dir, masks_dir, counts = synthetic
    cfg = load_yaml(ROOT / "configs/dataset.yaml")
    cfg["paths"].update(images_dir=str(images_dir), masks_dir=str(masks_dir), mask_suffix="")
    cfg["min_instance_area"] = 10
    cfg["processed_dir"] = str(tmp_path / "processed")
    cfg_path = tmp_path / "dataset.yaml"
    save_yaml(cfg, cfg_path)

    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--dataset", str(cfg_path)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr + res.stdout

    out = tmp_path / "processed"
    splits = read_json(out / "splits.json")
    assert sum(len(v) for v in splits.values()) == len(counts)
    assert (out / "yolo/data.yaml").exists()
    for split in ("train", "val", "test"):
        recs = load_split_records(out, split)
        assert [r.image_id for r in recs] == splits[split]
        assert all(r.count == counts[r.image_id] for r in recs)
        assert len(list((out / "yolo/labels" / split).glob("*.txt"))) == len(recs)
