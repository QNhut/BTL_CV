"""Opt-in disk quota guards. GB is decimal; no files are deleted by the guard."""
import os
import shutil
from pathlib import Path

GB = 10**9

# Conservative planning allowances, not measured checkpoint/VRAM sizes.
# Includes source cache, best/last/temp checkpoint and modest artifacts.
MODEL_DISK_GB = {'yolo': .8, 'csrnet': 1.8, 'faster_rcnn': 1.5, 'mask_rcnn': 1.7,
                 'rtdetrv2': 3.5, 'rf_detr': 2.0, 'grounding_dino': 1.5,
                 'countgd': 3.5, 'va_count': 2.0}


def directory_bytes(root):
    total, seen = 0, set()
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not Path(folder, name).is_symlink()]
        for name in files:
            path = Path(folder, name)
            if path.is_symlink():
                continue
            try:
                info = path.stat()
            except FileNotFoundError:
                continue  # A running logger may have replaced a file.
            key = (info.st_dev, info.st_ino)
            if key not in seen:
                total += info.st_size
                seen.add(key)
    return total


def check_disk(root, required_bytes=0, budget_gb=18, reserve_gb=2):
    root = Path(root).resolve()
    usage = shutil.disk_usage(root)
    used = directory_bytes(root)
    available = min(usage.free, int(budget_gb * GB) - used)
    needed = int(required_bytes + reserve_gb * GB)
    report = {'working_root': str(root), 'used_gb': used / GB,
              'physical_free_gb': usage.free / GB, 'budget_gb': budget_gb,
              'remaining_budget_gb': (budget_gb * GB - used) / GB,
              'additional_required_gb': required_bytes / GB, 'reserve_gb': reserve_gb}
    if needed > available:
        raise RuntimeError(f'Disk budget exceeded: need {needed / GB:.2f} GB including reserve, '
                           f'available {available / GB:.2f} GB under {root}. '
                           'Save outputs and run fewer models in a separate session; no checkpoint was deleted.')
    return report


def guard_disk(required_bytes=0):
    root = os.environ.get('FRUITCOUNT_WORKING_ROOT')
    if root:
        return check_disk(root, required_bytes,
                          float(os.environ.get('FRUITCOUNT_DISK_BUDGET_GB', '18')),
                          float(os.environ.get('FRUITCOUNT_MIN_FREE_GB', '2')))


def state_bytes(value):
    """Conservative tensor payload estimate for torch.save; no tensor copies."""
    if hasattr(value, 'numel') and hasattr(value, 'element_size'):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(state_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(state_bytes(item) for item in value)
    return 0
