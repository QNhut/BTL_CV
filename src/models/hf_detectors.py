"""Transformers detectors sharing the project's training and counting interfaces."""
from __future__ import annotations

import inspect
import numpy as np
import torch

from .base import CountResult
from .torch_base import TorchCounter


class HFDetectorCounter(TorchCounter):
    family = 'detection'
    dataset_kind = 'detection'

    def _build_network(self):
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as exc:
            raise ImportError('Install requirements-extra-models.txt for Transformers detectors.') from exc
        settings = self.cfg['model']
        source = settings['hub_id']
        options = {'revision': settings.get('revision', 'main'), 'trust_remote_code': False}
        self.grounded = self.cfg['type'] == 'grounding_dino'
        processor_class = AutoProcessor if self.grounded else AutoImageProcessor
        self.processor = processor_class.from_pretrained(source, **options)
        loader = AutoModelForZeroShotObjectDetection if self.grounded else AutoModelForObjectDetection
        if self.cfg.get('mode') == 'finetune' and not self.grounded:
            options.update(num_labels=1, id2label={0: 'fruit'}, label2id={'fruit': 0},
                           ignore_mismatched_sizes=True)
        network = loader.from_pretrained(source, **options)
        train = self.cfg.get('train', {})
        if (self.cfg.get('mode') == 'finetune' and train.get('scope') != 'full'
                and train.get('freeze_backbone', False)):
            for name, parameter in network.named_parameters():
                if 'backbone' in name:
                    parameter.requires_grad_(False)
        self.weight_files = []
        from huggingface_hub import try_to_load_from_cache
        from pathlib import Path
        import json
        for filename in ('model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json', 'pytorch_model.bin.index.json'):
            cached = str(Path(source) / filename) if Path(source).is_dir() else try_to_load_from_cache(source, filename, revision=options['revision'])
            if isinstance(cached, str) and Path(cached).is_file():
                if filename.endswith('.index.json'):
                    shards = set(json.loads(Path(cached).read_text())['weight_map'].values())
                    self.weight_files = [str(Path(cached).parent / shard) for shard in shards]
                else:
                    self.weight_files = [cached]
                break
        return network

    def _inputs(self, images, annotations=None):
        kwargs = {'images': images, 'return_tensors': 'pt'}
        if self.grounded:
            prompt = self.cfg['model'].get('prompt', '').strip().lower()
            if not prompt:
                raise ValueError('A target prompt is required, for example apple.')
            if not prompt.endswith('.'):
                prompt += '.'
            kwargs['text'] = [prompt] * len(images)
            kwargs['padding'] = True
        if annotations is not None and not self.grounded:
            kwargs['annotations'] = annotations
        inputs = self.processor(**kwargs)
        if annotations is not None and self.grounded:
            # GroundingDinoProcessor drops annotations; its image processor converts
            # COCO boxes to normalized cxcywh labels expected by the native loss.
            inputs['labels'] = self.processor.image_processor(
                images=images, annotations=annotations, return_tensors='pt')['labels']
        inputs = inputs.to(self.device)
        # BatchFeature.to() does not consistently recurse into label mappings
        # across Transformers versions. Move annotation tensors explicitly.
        if 'labels' in inputs:
            inputs['labels'] = [{key: value.to(self.device) if torch.is_tensor(value) else value
                                 for key, value in label.items()} for label in inputs['labels']]
        return inputs

    def training_step(self, batch):
        if self.cfg.get('mode') != 'finetune':
            raise ValueError('Set mode: finetune to train a Transformers detector.')
        images, targets = batch
        arrays, annotations = [], []
        for i, (image, target) in enumerate(zip(images, targets)):
            arrays.append(image.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy())
            objects = []
            for x0, y0, x1, y1 in target['boxes'].tolist():
                width, height = x1 - x0, y1 - y0
                objects.append({'bbox': [x0, y0, width, height], 'category_id': 0,
                                'area': width * height, 'iscrowd': 0})
            annotations.append({'image_id': i, 'annotations': objects})
        result = self.network(**self._inputs(arrays, annotations))
        if result.loss is None:
            raise RuntimeError('Detector returned no training loss; check annotation processing and Transformers version.')
        return {'loss': result.loss, **{f'loss_{key}': value.detach()
                                      for key, value in (result.loss_dict or {}).items()}}

    def _predict_raw(self, images):
        if not images:
            return []
        inputs = self._inputs(images)
        output = self.network(**inputs)
        sizes = torch.tensor([image.shape[:2] for image in images], device=self.device)
        if self.grounded:
            post = self.processor.post_process_grounded_object_detection
            threshold_key = 'threshold' if 'threshold' in inspect.signature(post).parameters else 'box_threshold'
            results = post(output, input_ids=inputs['input_ids'], target_sizes=sizes,
                           text_threshold=float(self.cfg['model'].get('text_threshold', 0.25)),
                           **{threshold_key: self.min_score})
        else:
            results = self.processor.post_process_object_detection(output, target_sizes=sizes,
                                                                   threshold=self.min_score)
        allowed = ({'fruit'} if self.cfg.get('mode') == 'finetune' else
                   {str(name).casefold() for name in self.cfg['model'].get('target_labels', [])})
        counts = []
        for result in results:
            boxes = result['boxes'].detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
            scores = result['scores'].detach().cpu().numpy().astype(np.float32)
            if allowed and not self.grounded:
                mapping = self.network.config.id2label
                names = [str(mapping.get(int(label), mapping.get(str(int(label)), ''))).casefold()
                         for label in result['labels']]
                keep = np.array([name in allowed for name in names], dtype=bool)
                boxes, scores = boxes[keep], scores[keep]
            counts.append(CountResult(boxes=boxes, scores=scores, score_thr=self.count_threshold))
        return counts
