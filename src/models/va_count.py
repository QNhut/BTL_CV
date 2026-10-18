"""VA-Count density checkpoint with automatic DINO exemplar selection.

This adapter labels its top-score exemplar policy explicitly; it does not claim
to reproduce the paper's separately trained CLIP binary exemplar classifier.
"""
import importlib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .base import CountResult
from .torch_base import TorchCounter
from .hf_detectors import HFDetectorCounter
from .research_utils import add_source_root, checkpoint_path, load_state


class VACountCounter(TorchCounter):
    family = 'density'
    supports_tiling = False

    def _build_network(self):
        settings = self.cfg['model']
        add_source_root(settings['source_dir'], 'models_mae_cross.py')
        module = importlib.import_module('models_mae_cross')
        # The upstream constructor passes qk_scale, removed by current timm.
        from timm.models.vision_transformer import Block
        import inspect
        if 'qk_scale' not in inspect.signature(Block).parameters:
            def compatible_block(*args, **kwargs):
                scale = kwargs.pop('qk_scale', None)
                if scale is not None:
                    raise ValueError('Non-default qk_scale needs upstream timm')
                return Block(*args, **kwargs)
            module.Block = compatible_block
        network = getattr(module, settings.get('architecture', 'mae_vit_base_patch16'))()
        weights = checkpoint_path(settings)
        network.load_state_dict(load_state(weights), strict=True)
        self.weight_files = [str(weights)]
        detector_cfg = {
            'name': 'va_count_exemplar_detector', 'type': 'grounding_dino', 'family': 'detection',
            'mode': 'pretrained',
            'model': {'hub_id': settings.get('exemplar_detector', 'IDEA-Research/grounding-dino-base'),
                      'prompt': settings.get('prompt', 'apple.'), 'text_threshold': 0.05},
            'inference': {'min_score': 0.05, 'count_threshold': 0.05},
        }
        self.proposer = HFDetectorCounter(detector_cfg, device=str(self.device))
        self.auxiliary_networks = [self.proposer.network]
        self.weight_files += self.proposer.weight_files
        return network

    def num_parameters(self):
        return super().num_parameters() + self.proposer.num_parameters()

    def training_step(self, batch):
        if self.cfg.get('mode') != 'finetune':
            raise ValueError('Set mode: finetune to train VA-Count.')
        from ..data.density import boxes_to_density
        import torch.nn.functional as F
        images, targets = batch
        losses = []
        for image, target in zip(images, targets):
            h, w = image.shape[-2:]
            boxes = target['boxes'].detach().cpu().numpy().copy()
            boxes[:, [0, 2]] *= 384 / w
            boxes[:, [1, 3]] *= 384 / h
            resized = TF.resize(image, [384, 384], antialias=True).to(self.device)
            patches = []
            # Train exemplars come only from augmented training annotations.
            for x0, y0, x1, y1 in boxes[:3]:
                left, top = max(0, int(x0)), max(0, int(y0))
                right, bottom = min(384, int(x1) + 1), min(384, int(y1) + 1)
                if left < right and top < bottom:
                    patches.append(TF.resize(resized[:, top:bottom, left:right], [64, 64], antialias=True))
            if not patches:
                # Empty crops remain valid negatives rather than dropping the batch.
                patches = [torch.zeros((3, 64, 64), device=self.device)]
            while len(patches) < 3:
                patches.append(patches[-1])
            raw = self.network(resized[None], torch.stack(patches)[None], 3)[0]
            density = boxes_to_density(boxes, (384, 384), downsample=1)
            expected = torch.as_tensor(density, device=self.device, dtype=raw.dtype).reshape_as(raw) * 60
            mse = F.mse_loss(raw, expected)
            count = (raw.sum() / 60 - len(boxes)).abs()
            losses.append(mse + float(self.train_cfg.get('count_loss_weight', 0.01)) * count)
        return {'loss': torch.stack(losses).mean()}

    def _predict_raw(self, images):
        results = []
        for image in images:
            proposal = self.proposer.predict_image(image)
            h, w = image.shape[:2]
            patches = []
            resized = TF.to_tensor(Image.fromarray(image).resize((384, 384), Image.Resampling.BILINEAR))
            for index in np.argsort(-proposal.scores):
                x0, y0, x1, y1 = proposal.boxes[index]
                if not (5 <= x1 - x0 <= w / 2 and 5 <= y1 - y0 <= h / 2):
                    continue
                left, top = max(0, int(x0 * 384 / w)), max(0, int(y0 * 384 / h))
                right, bottom = min(384, int(x1 * 384 / w) + 1), min(384, int(y1 * 384 / h) + 1)
                if left < right and top < bottom:
                    patches.append(TF.resize(resized[:, top:bottom, left:right], [64, 64]))
                if len(patches) == 3:
                    break
            if not patches:
                density = np.zeros((384, 384), np.float32)
            else:
                while len(patches) < 3:
                    patches.append(patches[-1])
                exemplars = torch.stack(patches)[None].to(self.device)
                raw = self.network(resized[None].to(self.device), exemplars, 3)[0]
                # Match upstream abs(sum(output))/60, without rounding counts.
                density = (raw * (-1 if raw.sum() < 0 else 1) / 60).cpu().numpy().astype(np.float32)
            results.append(CountResult(density=density, density_count=float(density.sum())))
        return results

    def load(self, weights):
        self.network.load_state_dict(load_state(weights), strict=True)
        self.network.eval()
        self.weight_files = [str(weights)]
