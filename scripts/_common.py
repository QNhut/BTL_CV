"""Tiện ích dùng chung cho các script CLI (thêm thư mục gốc project vào sys.path)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# CLI workers save figures to files. Do not inherit Jupyter's inline backend:
# isolated model environments may not contain matplotlib_inline/IPython.
os.environ['MPLBACKEND'] = 'Agg'

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")


def run_dir(out_root: str | Path, model_cfg: dict) -> Path:
    """Mỗi model có một thư mục kết quả riêng: <out_root>/<tên model>."""
    return Path(out_root) / model_cfg["name"]


def default_weights(out_root: str | Path, model_cfg: dict) -> Path:
    return run_dir(out_root, model_cfg) / "weights" / "best.pt"


def selected_weights(out_root, model_cfg, explicit=None):
    if explicit:
        return Path(explicit)
    if model_cfg.get('mode') == 'pretrained':
        return None  # The adapter loads its configured source checkpoint.
    return default_weights(out_root, model_cfg)
