import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from restore_runs import restore_run
from src.utils import save_yaml


def test_restore_preserves_history_and_existing_local_progress(tmp_path):
    source, target = tmp_path / 'input/model', tmp_path / 'working/model'
    (source / 'weights').mkdir(parents=True)
    (source / 'weights/last.pt').write_bytes(b'optimizer state')
    (source / 'weights/best.pt').write_bytes(b'best weights')
    (source / 'training').mkdir()
    (source / 'training/history.json').write_text('[]')
    cfg = {'name': 'model', 'type': 'countgd', 'mode': 'finetune', 'train': {'scope': 'decoder_heads'}}
    save_yaml(cfg, source / 'config.yaml')
    restore_run(source, target, cfg)
    assert (target / 'weights/last.pt').read_bytes() == b'optimizer state'
    assert (target / 'weights/best.pt').read_bytes() == b'best weights'
    assert (target / 'training/history.json').read_text() == '[]'
    (target / 'weights/last.pt').write_bytes(b'new local epoch')
    restore_run(source, target, cfg)
    assert (target / 'weights/last.pt').read_bytes() == b'new local epoch'


def test_restore_rejects_mismatched_optimizer_scope(tmp_path):
    source = tmp_path / 'input'
    (source / 'weights').mkdir(parents=True)
    (source / 'weights/last.pt').write_bytes(b'checkpoint')
    cfg = {'name': 'model', 'type': 'countgd', 'mode': 'finetune', 'train': {'scope': 'full'}}
    save_yaml(cfg, source / 'config.yaml')
    with pytest.raises(ValueError, match='scope differs'):
        restore_run(source, tmp_path / 'working', {**cfg, 'train': {'scope': 'decoder_heads'}})
    assert not (tmp_path / 'working/weights/last.pt').exists()


def test_best_only_is_not_a_resume_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match='last.pt'):
        restore_run(tmp_path / 'input', tmp_path / 'working', {'name': 'model'})


def test_report_restore_accepts_best_only_and_preserves_metrics(tmp_path):
    source, target = tmp_path / 'input/model', tmp_path / 'working/model'
    (source / 'weights').mkdir(parents=True)
    (source / 'weights/best.pt').write_bytes(b'best')
    (source / 'evaluation').mkdir()
    (source / 'evaluation/metrics_test.json').write_text('{"mae": 1}')
    restore_run(source, target, {'name': 'model'}, report_only=True)
    assert (target / 'weights/best.pt').read_bytes() == b'best'
    assert (target / 'evaluation/metrics_test.json').is_file()
    assert not (target / 'weights/last.pt').exists()
    (target / 'weights/best.pt').write_bytes(b'local')
    (source / 'evaluation/metrics_test.json').write_text('{"mae": 99}')
    restore_run(source, target, {'name': 'model'}, report_only=True)
    assert (target / 'weights/best.pt').read_bytes() == b'local'
    assert (target / 'evaluation/metrics_test.json').read_text() == '{"mae": 1}'


def test_report_restore_accepts_pretrained_metrics_without_weights(tmp_path):
    source, target = tmp_path / 'input/model', tmp_path / 'working/model'
    source.mkdir(parents=True)
    (source / 'metrics_test.json').write_text('{}')
    restore_run(source, target, {'name': 'model'}, report_only=True)
    assert (target / 'metrics_test.json').is_file()
