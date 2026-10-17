"""Group existing run artifacts: python scripts/organize_runs.py --runs /kaggle/working/runs."""
import argparse
from pathlib import Path
import shutil

from _common import ROOT
from src.utils.run_layout import organize_training_outputs


def organize_runs(runs):
    for model_dir in sorted(Path(runs).iterdir()):
        if not model_dir.is_dir() or model_dir.name == 'comparison':
            continue
        for pattern in ('metrics_*.json', 'per_image_*.csv', 'error_by_count_*.csv', 'pred_vs_gt_*.png'):
            for source in model_dir.glob(pattern):
                if not source.is_file():
                    continue
                destination = model_dir / 'evaluation' / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    print(f'Skipped existing destination: {destination}')
                    continue
                shutil.move(str(source), str(destination))
        organize_training_outputs(model_dir)
        print(f'Organized: {model_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default=str(ROOT / 'runs'))
    organize_runs(parser.parse_args().runs)
