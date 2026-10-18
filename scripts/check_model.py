"""Load a selected real checkpoint and test inference/backward on a few validation images."""
import argparse
import time
import numpy as np
import torch

from _common import ROOT, run_dir, selected_weights
from src.data import load_split_records
from src.data.dataset import build_dataset, get_collate_fn
from src.models import build_model, load_model_config
from src.utils import load_yaml, load_image, write_json
from src.utils.resources import guard_disk, MODEL_DISK_GB, GB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default=str(ROOT / 'configs/dataset.yaml'))
    parser.add_argument('--out', default=str(ROOT / 'runs'))
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    cfg = load_model_config(args.model)
    guard_disk(MODEL_DISK_GB.get(cfg['type'], 2) * GB)
    records = load_split_records(ROOT / load_yaml(args.dataset)['processed_dir'], 'val')
    if not records:
        raise ValueError('A non-empty validation split is required for the smoke check.')
    started = time.perf_counter()
    model = build_model(cfg)
    weights = selected_weights(args.out, cfg)
    if weights is not None and weights.is_file():
        model.load(str(weights))
    result = model.predict_image(load_image(records[0].image_path))
    if not np.isfinite(result.count):
        raise ValueError('Smoke-test count is not finite.')
    report = {'model': cfg['name'], 'count': result.count, 'protocol': cfg.get('protocol'),
              'inference_passed': True, 'backward_passed': False}
    if args.backward and cfg.get('mode') == 'finetune':
        model.train_mode()
        train_records = load_split_records(ROOT / load_yaml(args.dataset)['processed_dir'], 'train')
        if not train_records:
            raise ValueError('Backward smoke check requires a non-empty training split.')
        dataset = build_dataset(model.dataset_kind, train_records[:1])
        batch = get_collate_fn(model.dataset_kind)([dataset[0]])
        settings = cfg.get('train', {})
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=settings.get('lr', 1e-5),
                                      weight_decay=settings.get('weight_decay', 0))
        amp = bool(settings.get('amp', True)) and model.device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        with torch.autocast(device_type=model.device.type, enabled=amp):
            losses = model.training_step(batch)
        if not torch.isfinite(losses['loss']):
            raise ValueError('Smoke-test loss is not finite.')
        scaler.scale(losses['loss']).backward()
        if settings.get('grad_clip', 0):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), settings['grad_clip'])
        scaler.step(optimizer)  # Allocate real optimizer states; no checkpoint is overwritten.
        scaler.update()
        report.update(backward_passed=True, loss=float(losses['loss'].detach()), amp=amp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        report['gpu_peak_allocated_gb'] = torch.cuda.max_memory_allocated() / GB
    report['elapsed_s'] = time.perf_counter() - started
    write_json(report, run_dir(args.out, cfg) / 'evaluation/smoke_check.json')
    print(report)


if __name__ == '__main__':
    main()
