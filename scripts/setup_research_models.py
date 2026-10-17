"""Fetch upstream research source, without downloading weights or changing packages."""
import argparse
import json
from pathlib import Path

from _common import ROOT


SOURCES = {
    'countgd': ('nikigoli/countgd', 'space'),
    'va_count': ('HopooLinZ/VA-Count', 'model'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(SOURCES), required=True)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download, HfApi
    repo_id, repo_type = SOURCES[args.model]
    revision = HfApi().repo_info(repo_id, repo_type=repo_type, revision=args.revision).sha
    destination = Path(args.out) if args.out else ROOT / 'external' / args.model
    snapshot_download(repo_id, repo_type=repo_type, revision=revision, local_dir=str(destination),
                      allow_patterns=['*.py', '*.cpp', '*.cu', '*.h', '*.cuh', '*.sh', 'requirements.txt', 'LICENSE*'])
    (destination / 'source_revision.json').write_text(json.dumps(
        {'repo_id': repo_id, 'repo_type': repo_type, 'revision': revision}, indent=2), encoding='utf-8')
    print(f'Source ready: {destination} at {revision}')
    if args.model == 'countgd':
        print('CountGD additionally needs upstream dependencies/CUDA ops. See Docs/Extra_Models.md.')


if __name__ == '__main__':
    main()
