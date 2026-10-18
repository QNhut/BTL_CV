"""Partial training must freeze feature extractors while retaining decoder gradients."""
import pytest
torch = pytest.importorskip('torch')
from src.models.countgd import CountGDCounter


@pytest.mark.parametrize('scope', ['decoder_heads', 'full'])
def test_countgd_parameter_selection(scope):
    counter = object.__new__(CountGDCounter)
    counter.cfg = {'type': 'countgd', 'mode': 'finetune', 'train': {'scope': scope}}
    counter.network = torch.nn.ModuleDict({
        'backbone': torch.nn.Linear(2, 2),
        'bert': torch.nn.Linear(2, 2),
        'transformer': torch.nn.ModuleDict({'decoder': torch.nn.Linear(2, 2)}),
        'class_embed': torch.nn.Linear(2, 1),
    })
    counter._apply_train_scope()
    for name, parameter in counter.network.named_parameters():
        assert parameter.requires_grad == (scope == 'full' or 'decoder' in name or 'class_embed' in name)
    counter.train_mode()
    assert counter.network['backbone'].training == (scope == 'full')
    assert counter.network['transformer']['decoder'].training
