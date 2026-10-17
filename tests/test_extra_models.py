"""Contract tests: no pretrained downloads or external CUDA builds required."""
from types import SimpleNamespace
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from src.models import build_model, available_models
from src.models.hf_detectors import HFDetectorCounter
from src.models.countgd import CountGDCounter
from src.models.va_count import VACountCounter
from src.models.base import CountResult
from src.utils import save_yaml


def test_new_models_registered():
    assert {'rtdetrv2', 'rf_detr', 'countgd', 'grounding_dino', 'va_count'} <= set(available_models())


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return Inputs(pixel_values=torch.zeros(1, 3, 16, 16), input_ids=torch.ones(1, 2, dtype=torch.long))

    def post_process_object_detection(self, outputs, target_sizes, threshold):
        self.sizes = target_sizes
        return [{'boxes': torch.tensor([[1., 2., 8., 9.], [2., 2., 6., 6.]]),
                 'scores': torch.tensor([0.9, 0.8]), 'labels': torch.tensor([0, 1])}]

    def post_process_grounded_object_detection(self, outputs, input_ids, target_sizes, threshold, text_threshold):
        self.threshold = threshold
        return self.post_process_object_detection(outputs, target_sizes, threshold)


class FakeNetwork(torch.nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(id2label=labels)

    def forward(self, **kwargs):
        return SimpleNamespace(loss=self.weight.square(), loss_dict={'bbox': self.weight.square()})


def configure_fake_hf(self):
    self.processor = Processor()
    self.grounded = self.cfg['type'] == 'grounding_dino'
    return FakeNetwork({0: 'fruit' if self.cfg.get('mode') == 'finetune' else 'Apple', 1: 'Potato'})


def test_hf_filters_target_class_and_preserves_image_coordinates(monkeypatch):
    monkeypatch.setattr(HFDetectorCounter, '_build_network', configure_fake_hf)
    model = build_model({'type': 'rtdetrv2', 'mode': 'pretrained', 'model': {'target_labels': ['apple']}})
    result = model.predict_image(np.zeros((20, 30, 3), np.uint8))
    assert result.count == 1
    assert result.boxes.tolist() == [[1., 2., 8., 9.]]
    assert model.processor.sizes.tolist() == [[20, 30]]


def test_hf_training_labels_are_zero_based_coco_and_loss_has_grad(monkeypatch):
    monkeypatch.setattr(HFDetectorCounter, '_build_network', configure_fake_hf)
    model = build_model({'type': 'rf_detr', 'mode': 'finetune', 'model': {}})
    losses = model.training_step(([torch.zeros(3, 20, 30)], [{'boxes': torch.tensor([[2., 3., 8., 10.]])}]))
    annotation = model.processor.kwargs['annotations'][0]['annotations'][0]
    assert annotation['category_id'] == 0 and annotation['bbox'] == [2., 3., 6., 7.]
    losses['loss'].backward()
    assert model.network.weight.grad is not None
    assert model.predict_image(np.zeros((20, 30, 3), np.uint8)).count == 1


def test_grounding_prompt_is_normalized_and_threshold_is_forwarded(monkeypatch):
    monkeypatch.setattr(HFDetectorCounter, '_build_network', configure_fake_hf)
    model = build_model({'type': 'grounding_dino', 'model': {'prompt': 'Apple'},
                         'inference': {'min_score': 0.15}})
    model.predict_image(np.zeros((20, 30, 3), np.uint8))
    assert model.processor.kwargs['text'] == ['apple.']
    assert model.processor.threshold == 0.15


def test_hf_annotation_tensors_are_explicitly_moved_to_device(monkeypatch):
    monkeypatch.setattr(HFDetectorCounter, '_build_network', configure_fake_hf)
    model = build_model({'type': 'rf_detr', 'mode': 'finetune', 'model': {}})
    model.device = torch.device('meta')
    model.processor = lambda **kwargs: Inputs(labels=[{'class_labels': torch.tensor([0]),
                                                       'boxes': torch.zeros(1, 4)}])
    inputs = model._inputs([np.zeros((16, 16, 3), np.uint8)], annotations=[])
    assert inputs['labels'][0]['class_labels'].device.type == 'meta'
    assert inputs['labels'][0]['boxes'].device.type == 'meta'


def test_countgd_converts_normalized_boxes_without_claiming_box_map(monkeypatch):
    class Network(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return {'pred_logits': torch.tensor([[[3.0], [-3.0]]]),
                    'pred_boxes': torch.tensor([[[.5, .5, .2, .4], [.2, .2, .1, .1]]])}

    def build(self):
        self.nested_tensor = lambda images: images
        return Network()

    monkeypatch.setattr(CountGDCounter, '_build_network', build)
    model = build_model({'type': 'countgd', 'model': {'prompt': 'apple'}, 'inference': {'min_score': .1}})
    result = model.predict_image(np.zeros((100, 200, 3), np.uint8))
    assert result.count == 1
    assert result.boxes[0] == pytest.approx([80, 30, 120, 70])
    assert not model.supports_box_metrics


def test_va_density_scale_and_automatic_exemplars(monkeypatch):
    class Network(torch.nn.Module):
        def forward(self, images, patches, shots):
            assert tuple(images.shape) == (1, 3, 384, 384)
            assert tuple(patches.shape) == (1, 3, 3, 64, 64) and shots == 3
            return torch.full((1, 384, 384), 120 / (384 * 384))

    def build(self):
        self.proposer = SimpleNamespace(predict_image=lambda image: CountResult(
            boxes=np.array([[10, 10, 30, 30]], np.float32), scores=np.array([.9], np.float32)))
        return Network()

    monkeypatch.setattr(VACountCounter, '_build_network', build)
    model = build_model({'type': 'va_count', 'model': {}})
    result = model.predict_image(np.zeros((100, 100, 3), np.uint8))
    assert result.is_density and result.count == pytest.approx(2, abs=1e-4)
    assert result.density.sum() == pytest.approx(result.count)


def test_pretrained_train_script_skips_without_optional_packages(tmp_path):
    cfg = tmp_path / 'grounding.yaml'
    save_yaml({'name': 'grounding', 'type': 'grounding_dino', 'family': 'detection',
               'mode': 'pretrained', 'model': {'hub_id': 'does-not-need-downloading'}}, cfg)
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / 'scripts/train.py'), '--model', str(cfg),
                                '--out', str(tmp_path / 'runs')], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / 'runs/grounding/config.yaml').exists()
