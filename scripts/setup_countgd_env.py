# CountGD uses its own CUDA compiler and Python; the notebook kernel stays unchanged.
import io
import tarfile
import urllib.request

from _common import ROOT
from src.utils import load_yaml, save_yaml
import argparse
import os
import subprocess
from pathlib import Path
WORK = str(ROOT)
WORKING_ROOT = Path(os.environ.get('FRUITCOUNT_WORKING_ROOT', '/kaggle/working'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    args = parser.parse_args()
    from src.utils.resources import guard_disk, GB
    guard_disk(8 * GB)
    tools = WORKING_ROOT / '.tools'
    tools.mkdir(exist_ok=True)
    mamba = tools / 'micromamba'
    if not mamba.is_file():
        url = 'https://micro.mamba.pm/api/micromamba/linux-64/latest'
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = response.read()
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:bz2') as bundle:
            member = bundle.extractfile('bin/micromamba')
            if member is None:
                raise RuntimeError('micromamba archive has no executable')
            mamba.write_bytes(member.read())
        mamba.chmod(0o755)
    prefix = Path(args.env).resolve()
    env = dict(os.environ)
    env['MAMBA_ROOT_PREFIX'] = str(WORKING_ROOT / '.mamba')
    # Install the compiler, headers and CUDA runtime development files, not a GPU driver.
    # https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-linux/
    subprocess.run([
        str(mamba), 'create', '-y', '-p', str(prefix), '--override-channels',
        '-c', 'nvidia/label/cuda-12.4.1', '-c', 'conda-forge',
        'python=3.10', 'pip', 'cuda-nvcc=12.4', 'cuda-cudart-dev=12.4',
        'cuda-cccl=12.4', 'gxx_linux-64=12',
    ], env=env, check=True)
    subprocess.run([str(mamba), 'clean', '--all', '-y'], env=env, check=True)
    python = prefix / 'bin/python'
    subprocess.run([
        str(python), '-m', 'pip', 'install', '--no-cache-dir',
        'torch==2.5.1', 'torchvision==0.20.1',
        '--index-url', 'https://download.pytorch.org/whl/cu124',
    ], check=True)
    subprocess.run([
        str(python), '-m', 'pip', 'install', '--no-cache-dir',
        'setuptools<81', 'wheel', 'ninja', '-r', str(Path(WORK) / 'requirements-countgd.txt'),
    ], check=True)
    env.update(CUDA_HOME=str(prefix), CUDA_PATH=str(prefix), MAX_JOBS='1',
               CC=str(prefix / 'bin/x86_64-conda-linux-gnu-gcc'),
               CXX=str(prefix / 'bin/x86_64-conda-linux-gnu-g++'))
    env['CUDAHOSTCXX'] = env['CXX']
    env['PATH'] = str(prefix / 'bin') + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = os.pathsep.join([
        str(prefix / 'lib'), str(prefix / 'targets/x86_64-linux/lib'),
        env.get('LD_LIBRARY_PATH', ''),
    ])
    subprocess.run([str(prefix / 'bin/nvcc'), '--version'], env=env, check=True)
    subprocess.run([
        str(python), '-c',
        'import torch; print(torch.__version__, torch.version.cuda); '
        'assert torch.version.cuda == "12.4"; '
        'assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator"',
    ], env=env, check=True)
    ops = Path(WORK) / 'external/countgd/models/GroundingDINO/ops'
    from countgd_ops_compat import patch_ops
    patch_ops(ops)
    # Rebuild after migrating upstream CUDA APIs.
    from countgd_ops_compat import build_ops
    build_ops(python, ops, env, WORKING_ROOT / 'countgd_build.log')
    subprocess.run([str(python), '-c', 'import torch, MultiScaleDeformableAttention'],
                   cwd=WORK, env=env, check=True)
    cfg = load_yaml(ROOT / 'configs/models/countgd.yaml')
    cfg['python_executable'] = str(python)
    save_yaml(cfg, ROOT / 'configs/models/countgd.yaml')

    print('CountGD ready:', python)

if __name__ == '__main__':
    main()
