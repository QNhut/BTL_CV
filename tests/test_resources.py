import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.resources import check_disk, directory_bytes


def test_budget_applies_even_when_physical_disk_has_plenty_free(tmp_path):
    (tmp_path / 'cache.bin').write_bytes(b'0' * 100)
    assert directory_bytes(tmp_path) == 100
    report = check_disk(tmp_path, required_bytes=50, budget_gb=200 / 10**9, reserve_gb=0)
    assert report['remaining_budget_gb'] == pytest.approx(100 / 10**9)
    with pytest.raises(RuntimeError, match='Disk budget exceeded'):
        check_disk(tmp_path, required_bytes=101, budget_gb=200 / 10**9, reserve_gb=0)


def test_reserve_is_kept_in_addition_to_requested_write(tmp_path):
    with pytest.raises(RuntimeError):
        check_disk(tmp_path, required_bytes=80, budget_gb=100 / 10**9, reserve_gb=30 / 10**9)


def test_checkpoint_guard_keeps_existing_last_on_budget_failure(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from src.training import checkpoint
    last = tmp_path / 'last.pt'
    last.write_bytes(b'old checkpoint')
    def fail(*args):
        raise RuntimeError('Disk budget exceeded')
    monkeypatch.setattr(checkpoint, 'guard_disk', fail)
    with pytest.raises(RuntimeError):
        checkpoint.save_checkpoint(last, {'epoch': 1})
    assert last.read_bytes() == b'old checkpoint'
    assert not (tmp_path / 'last.pt.tmp').exists()


def test_checkpoint_io_failure_removes_only_own_partial_temp(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from src.training import checkpoint
    last = tmp_path / 'last.pt'
    last.write_bytes(b'old checkpoint')
    monkeypatch.setattr(checkpoint, 'guard_disk', lambda *args: None)
    def fail(state, destination):
        Path(destination).write_bytes(b'partial')
        raise OSError('disk full')
    monkeypatch.setattr(checkpoint.torch, 'save', fail)
    with pytest.raises(OSError):
        checkpoint.save_checkpoint(last, {})
    assert last.read_bytes() == b'old checkpoint'
    assert not (tmp_path / 'last.pt.tmp').exists()


def test_combined_plan_rejects_ten_models_under_18gb(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / 'scripts/check_resources.py'),
        '--models', 'yolov8', 'yolo11', 'csrnet', 'faster_rcnn', 'mask_rcnn',
        'rtdetrv2', 'rf_detr', 'grounding_dino', 'countgd', 'va_count',
        '--working-root', str(tmp_path)], text=True, capture_output=True)
    assert completed.returncode != 0
    report = json.loads((tmp_path / 'runs/resource_plan.json').read_text())
    assert 'disk_error' in report and report['estimate_only']
