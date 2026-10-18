"""Restore selected runs from read-only Kaggle Input without overwriting local progress."""
import argparse
from pathlib import Path
import shutil

from _common import ROOT
from src.utils import load_yaml
from src.utils.resources import guard_disk


def restore_run(source, destination, cfg):
    source, destination = Path(source), Path(destination)
    if (destination / 'weights/last.pt').is_file():
        return f"{cfg['name']}: giữ checkpoint hiện có trong working"
    if not (source / 'weights/last.pt').is_file():
        raise FileNotFoundError(f'Missing resume checkpoint: {source / "weights/last.pt"}')
    saved_config = source / 'config.yaml'
    if saved_config.is_file():
        saved = load_yaml(saved_config)
        for key in ('type', 'mode'):
            if saved.get(key) != cfg.get(key):
                raise ValueError(f'{source}: {key} differs from current config; cannot resume safely')
        if saved.get('train', {}).get('scope', 'default') != cfg.get('train', {}).get('scope', 'default'):
            raise ValueError(f'{source}: train.scope differs; use a matching config to resume optimizer')
    # Restore weights, histories and provenance, not large prediction/plot folders.
    files = [p for p in (source / 'weights').glob('*.pt') if p.is_file()]
    files += [source / name for name in ('config.yaml', 'args.yaml', 'results.csv')
              if (source / name).is_file()]
    files += [p for p in (source / 'training').glob('*')
              if p.is_file() and p.suffix in ('.json', '.csv', '.yaml')]
    guard_disk(sum(p.stat().st_size for p in files))
    for path in sorted(files, key=lambda p: p.name == 'last.pt'):
        target = destination / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Publish last.pt atomically so an interrupted copy is never resumed.
        temporary = target.with_name(target.name + '.restoring')
        shutil.copy2(path, temporary)
        temporary.replace(target)
    return f"{cfg['name']}: restored {source} -> {destination}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    args = parser.parse_args()
    root = Path(args.source)
    if not root.is_dir():
        raise FileNotFoundError(f'Resume source does not exist: {root}')
    if (root / 'runs').is_dir():
        root = root / 'runs'
    for key in args.models:
        cfg = load_yaml(ROOT / 'configs/models' / f'{key}.yaml')
        if cfg.get('mode') == 'pretrained':
            continue
        source = root / cfg['name']
        if not source.is_dir():
            print(f"{key}: không có run {source}; dùng checkpoint working hoặc train mới")
            continue
        print(restore_run(source, Path(args.out) / cfg['name'], cfg))


if __name__ == '__main__':
    main()
