"""Training contract without legacy CUDA dependencies; production smoke runs on Kaggle."""
import pytest
torch = pytest.importorskip('torch')
from src.models.countgd import CountGDCounter


def test_countgd_normalizes_boxes_and_uses_upstream_weights():
    counter = object.__new__(CountGDCounter)
    counter.cfg = {'mode': 'finetune', 'model': {'prompt': 'apple'}}
    counter.device = torch.device('cpu')
    counter.nested_tensor = lambda tensors: tensors

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.))
        def forward(self, images, exemplar_images, boxes, labels, **kwargs):
            assert kwargs['captions'] == ['apple .']
            assert len(boxes) == 1 and boxes[0].shape == (0, 4)
            return {'pred_logits': self.weight.reshape(1, 1, 1)}

    class Criterion(torch.nn.Module):
        weight_dict = {'loss_ce': 2., 'loss_bbox': 5.}
        def forward(self, outputs, targets, categories, captions):
            torch.testing.assert_close(targets[0]['boxes'], torch.tensor([[.5, .5, .6, .4]]))
            assert targets[0]['labels'].tolist() == [0]
            assert categories == [['apple']]
            value = outputs['pred_logits'].sum()
            return {'loss_ce': value, 'loss_bbox': value * 2, 'diagnostic': value * 99}

    counter.network = Network()
    counter.criterion = Criterion()
    losses = counter.training_step(([torch.rand(3, 100, 200)],
                                    [{'boxes': torch.tensor([[40., 30., 160., 70.]])}]))
    torch.testing.assert_close(losses['loss'], torch.tensor(24.))
    losses['loss'].backward()
    torch.testing.assert_close(counter.network.weight.grad, torch.tensor(12.))
