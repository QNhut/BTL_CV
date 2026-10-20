"""Discover configured models with artifacts from previous notebook sessions."""
from pathlib import Path

from src.utils import load_yaml


def runs_root(path):
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f'Run source does not exist: {root}')
    return root / 'runs' if (root / 'runs').is_dir() else root


def discover_models(config_dir, roots):
    roots = [runs_root(root) for root in roots]
    keys = []
    for config in sorted(Path(config_dir).glob('*.yaml')):
        cfg = load_yaml(config)
        name = cfg.get('name')
        if name and any((root / name).is_dir() for root in roots):
            keys.append(config.stem)
    return keys


def test_report(run):
    """Return the base test report, preferring the current grouped layout."""
    root = Path(run)
    return next((p for p in (root / 'evaluation/metrics_test.json', root / 'metrics_test.json')
                 if p.is_file()), None)
