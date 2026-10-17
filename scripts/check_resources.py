"""Preflight selected model dependencies and conservative disk planning; no weight downloads."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from _common import ROOT
from src.utils import load_yaml, write_json
from src.utils.resources import MODEL_DISK_GB, GB, check_disk


IMPORT_CHECK = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import numpy as np
import torch
cfg = json.loads(sys.argv[1])
kind = cfg['type']
if kind in ('faster_rcnn', 'mask_rcnn'):
    from torchvision.ops import nms
    nms(torch.zeros((1, 4)), torch.ones(1), .5)
if kind == 'yolo':
    from ultralytics import YOLO
if kind in ('rtdetrv2', 'rf_detr', 'grounding_dino', 'va_count'):
    import transformers
    from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor
    required = {'rtdetrv2': 'RTDetrV2ForObjectDetection', 'rf_detr': 'RfDetrForObjectDetection',
                'grounding_dino': 'GroundingDinoForObjectDetection', 'va_count': 'GroundingDinoForObjectDetection'}[kind]
    getattr(transformers, required)
if kind in ('countgd', 'va_count'):
    from src.models.research_utils import add_source_root
    import timm
    required = 'cfg_app.py' if kind == 'countgd' else 'models_mae_cross.py'
    add_source_root(cfg['model']['source_dir'], required)
if kind == 'va_count':
    import models_mae_cross
if kind == 'countgd':
    import transformers
    if int(transformers.__version__.split('.')[0]) >= 5:
        raise RuntimeError('CountGD needs its separate Transformers 4.x environment; set python_executable.')
    import MultiScaleDeformableAttention
    from util.slconfig import SLConfig
    import models
    if not torch.cuda.is_available():
        raise RuntimeError('The configured CountGD CUDA backend needs an available CUDA GPU.')
info = {'python': sys.executable, 'torch': str(torch.__version__), 'numpy': str(np.__version__), 'cuda': torch.cuda.is_available()}
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    info.update(gpu=torch.cuda.get_device_name(), gpu_free_gb=free/10**9, gpu_total_gb=total/10**9)
try:
    import psutil
    info['ram_available_gb'] = psutil.virtual_memory().available / 10**9
except ImportError:
    pass
print('RESOURCE_INFO=' + json.dumps(info))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True, help='Config keys, e.g. yolov8 rf_detr')
    parser.add_argument('--working-root', required=True)
    parser.add_argument('--budget-gb', type=float, default=18)
    parser.add_argument('--reserve-gb', type=float, default=2)
    parser.add_argument('--out', default=None)
    parser.add_argument('--check-imports', action='store_true')
    args = parser.parse_args()
    entries, errors = [], []
    for key in args.models:
        cfg = load_yaml(ROOT / 'configs/models' / f'{key}.yaml')
        entry = {'key': key, 'name': cfg['name'], 'type': cfg['type'],
                 'additional_disk_allowance_gb': MODEL_DISK_GB.get(cfg['type'], 2)}
        if args.check_imports:
            executable = cfg.get('python_executable') or sys.executable
            try:
                completed = subprocess.run([executable, '-c', IMPORT_CHECK, json.dumps(cfg), str(ROOT)],
                                           cwd=ROOT, text=True, capture_output=True)
                if completed.returncode:
                    entry['dependency_error'] = completed.stderr[-4000:]
                    errors.append(f"{key}: {entry['dependency_error']}")
                else:
                    metadata = next(line for line in completed.stdout.splitlines() if line.startswith('RESOURCE_INFO='))
                    entry['environment'] = json.loads(metadata.split('=', 1)[1])
            except (OSError, StopIteration, ValueError) as exc:
                entry['dependency_error'] = str(exc)
                errors.append(f'{key}: {exc}')
        entries.append(entry)
    additional = sum(entry['additional_disk_allowance_gb'] for entry in entries)
    report = {'models': entries, 'estimate_only': True, 'total_additional_allowance_gb': additional}
    try:
        report['disk'] = check_disk(args.working_root, additional * GB, args.budget_gb, args.reserve_gb)
    except RuntimeError as exc:
        report['disk_error'] = str(exc)
        errors.append(str(exc))
    destination = Path(args.out) if args.out else Path(args.working_root) / 'runs/resource_plan.json'
    write_json(report, destination)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit('Preflight failed before training:\n' + '\n'.join(errors))


if __name__ == '__main__':
    main()
