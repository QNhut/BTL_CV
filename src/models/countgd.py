"""Text-only CountGD, using the author's Hugging Face Space model implementation.

Not the paper's SAM test-time normalization or exemplar/crop ensemble protocol.
"""
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .base import CountResult
from .torch_base import TorchCounter
from .research_utils import add_source_root, checkpoint_path, load_into


class CountGDCounter(TorchCounter):
    family = 'detection'
    supports_tiling = False
    supports_box_metrics = False  # CountGD optimizes counting/localization, not fruit box extent.

    def _build_network(self):
        import transformers
        if int(transformers.__version__.split('.')[0]) >= 5:
            raise RuntimeError('CountGD upstream requires Transformers 4.x. Use a separate Python environment '
                               'and set python_executable in countgd.yaml; see Docs/Extra_Models.md.')
        settings = self.cfg['model']
        root = add_source_root(settings['source_dir'], 'cfg_app.py')
        try:
            import MultiScaleDeformableAttention
        except ImportError as exc:
            raise ImportError('Build CountGD CUDA ops from external/countgd/models/GroundingDINO/ops '
                              'in the CountGD environment before evaluation.') from exc
        from util.slconfig import SLConfig
        from util.misc import nested_tensor_from_tensor_list
        import models  # Registers the author's model builder.
        from models.registry import MODULE_BUILD_FUNCS

        self.nested_tensor = nested_tensor_from_tensor_list
        args = argparse.Namespace(**SLConfig.fromfile(str(root / 'cfg_app.py'))._cfg_dict.to_dict())
        args.device = str(self.device)
        args.text_encoder_type = settings.get('text_encoder', 'bert-base-uncased')
        args.use_checkpoint = args.use_transformer_ckpt = False
        network, self.criterion, _ = MODULE_BUILD_FUNCS.get(args.modelname)(args)
        weights = checkpoint_path(settings)
        load_into(network, weights)
        if (self.cfg.get('mode') == 'finetune' and self.train_cfg.get('scope') != 'full'
                and self.train_cfg.get('freeze_backbone', False)):
            for name, parameter in network.named_parameters():
                if name.startswith(('backbone.', 'bert.')):
                    parameter.requires_grad_(False)
        self.weight_files = [str(weights)]
        return network

    def training_step(self, batch):
        if self.cfg.get('mode') != 'finetune':
            raise ValueError('Set mode: finetune to train CountGD.')
        from torchvision.ops import box_convert
        images, targets = batch
        tensors, labels = [], []
        prompt = self.cfg['model'].get('prompt', 'apple').strip(' .')
        captions = [prompt + ' .'] * len(images)
        for image, target in zip(images, targets):
            h, w = image.shape[-2:]
            scale = min(800 / min(h, w), 1333 / max(h, w))
            resized = TF.resize(image, [round(h * scale), round(w * scale)], antialias=True)
            tensors.append(TF.normalize(resized, [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]).to(self.device))
            boxes = target['boxes'].to(self.device)
            boxes = box_convert(boxes, 'xyxy', 'cxcywh') / boxes.new_tensor([w, h, w, h])
            labels.append({'boxes': boxes, 'labels': torch.zeros(len(boxes), dtype=torch.long,
                                                                 device=self.device)})
        nested = self.nested_tensor(tensors)
        outputs = self.network(nested, nested,
                               [torch.empty((0, 4), device=self.device) for _ in images],
                               [torch.zeros(1, dtype=torch.long, device=self.device) for _ in images],
                               captions=captions)
        self.criterion.to(self.device)
        losses = self.criterion(outputs, labels, [[prompt] for _ in images], captions)
        total = sum(value * self.criterion.weight_dict[key] for key, value in losses.items()
                    if key in self.criterion.weight_dict)
        return {'loss': total, **{key: value.detach() for key, value in losses.items()}}

    def _predict_raw(self, images):
        prompt = self.cfg['model'].get('prompt', '').strip()
        if not prompt:
            raise ValueError('CountGD requires model.prompt, for example apple')
        results = []
        for image in images:
            h, w = image.shape[:2]
            scale = min(800 / min(h, w), 1333 / max(h, w))
            pil = Image.fromarray(image).resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)
            tensor = TF.normalize(TF.to_tensor(pil), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(self.device)
            nested = self.nested_tensor([tensor])
            prediction = self.network(nested, nested, [torch.empty((0, 4), device=self.device)],
                                      [torch.zeros(1, dtype=torch.long, device=self.device)],
                                      captions=[prompt.rstrip(' .') + ' .'])
            scores = prediction['pred_logits'][0].sigmoid().amax(-1)
            keep = scores >= self.min_score
            boxes = prediction['pred_boxes'][0][keep]
            xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
            xyxy *= torch.tensor([w, h, w, h], device=self.device)
            results.append(CountResult(boxes=xyxy.cpu().numpy().astype(np.float32),
                                       scores=scores[keep].cpu().numpy().astype(np.float32),
                                       score_thr=self.count_threshold))
        return results

    def load(self, weights):
        load_into(self.network, weights)
        self.network.eval()
        self.weight_files = [str(weights)]
