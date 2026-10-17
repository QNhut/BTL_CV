"""Optional test against downloaded upstream VA source, without pretrained weights."""
import importlib
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('timm')

from src.models.va_count import VACountCounter
from src.models.base import CountResult
from src.models.research_utils import add_source_root


def test_upstream_density_network_forward_without_production_downloads(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'external/va_count'
    if not (source / 'models_mae_cross.py').exists():
        pytest.skip('Run setup_research_models.py --model va_count first')
    add_source_root(source, 'models_mae_cross.py')
    module = importlib.import_module('models_mae_cross')
    from timm.models.vision_transformer import Block

    def compatible_block(*args, **kwargs):
        kwargs.pop('qk_scale', None)
        return Block(*args, **kwargs)

    monkeypatch.setattr(module, 'Block', compatible_block)
    def tiny():
        return module.SupervisedMAE(embed_dim=64, depth=1, num_heads=4,
                                    decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=8)
    monkeypatch.setattr(module, 'test_tiny', tiny, raising=False)
    checkpoint = tmp_path / 'tiny.pth'
    torch.save({'model': tiny().state_dict()}, checkpoint)

    class Proposer:
        weight_files = []
        network = torch.nn.Linear(1, 1)
        def __init__(self, *args, **kwargs):
            pass
        def predict_image(self, image):
            return CountResult(boxes=np.array([[10, 10, 30, 30]], np.float32), scores=np.array([.9], np.float32))

    monkeypatch.setattr('src.models.va_count.HFDetectorCounter', Proposer)
    counter = VACountCounter({'model': {'source_dir': str(source), 'weights': str(checkpoint),
                                        'architecture': 'test_tiny', 'device': 'cpu'}})
    result = counter.predict_image(np.zeros((100, 100, 3), np.uint8))
    assert result.density.shape == (384, 384)
    assert np.isfinite(result.count) and result.count >= 0
