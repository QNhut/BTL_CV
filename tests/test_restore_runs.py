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
