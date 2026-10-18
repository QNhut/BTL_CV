"""Create an isolated Python 3.10 / Transformers 4.x environment and build CountGD ops."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from _common import ROOT
from src.utils import load_yaml, save_yaml
from src.utils.resources import guard_disk, GB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', required=True)
    args = parser.parse_args()
    guard_disk(6 * GB)  # Isolated CUDA PyTorch + compiler build artifacts, before downloads.
    destination = Path(args.env).resolve()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'uv'], check=True)
    python = destination / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.10', str(destination)], check=True)
    subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', str(python),
                    'pip', 'setuptools<81', 'wheel', '-r', str(ROOT / 'requirements-countgd.txt')], check=True)
    ops = ROOT / 'external/countgd/models/GroundingDINO/ops'
    build_env = dict(os.environ)
    build_env.setdefault('MAX_JOBS', '2')
    subprocess.run([str(python), 'setup.py', 'build', 'install'], cwd=ops, env=build_env, check=True)
    subprocess.run([str(python), '-c', 'import torch, MultiScaleDeformableAttention; '
                    'assert torch.cuda.is_available(), "CountGD requires a CUDA GPU"'], check=True)
    config_path = ROOT / 'configs/models/countgd.yaml'
    cfg = load_yaml(config_path)
    cfg['python_executable'] = str(python)
    save_yaml(cfg, config_path)
    print(f'CountGD environment ready: {python}')


if __name__ == '__main__':
    main()
