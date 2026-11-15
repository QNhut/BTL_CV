"""Compatibility migrations and visible compiler diagnostics for CountGD ops."""
import re
import subprocess


def patch_ops(ops):
    if not (ops / 'setup.py').is_file():
        raise FileNotFoundError(f'Fetch CountGD source before building: {ops}')
    for path in (ops / 'src').rglob('*'):
        if path.suffix not in {'.h', '.cuh', '.cpp', '.cu'}:
            continue
        old = path.read_text(encoding='utf-8')
        new = old.replace('THC/THCAtomics.cuh', 'ATen/cuda/Atomic.cuh')
        new = re.sub(r'(\w+)\.type\(\)\.is_cuda\(\)', r'\1.is_cuda()', new)
        new = new.replace('AT_DISPATCH_FLOATING_TYPES(value.type(),',
                          'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),')
        new = new.replace('.data<', '.data_ptr<')
        if new != old:
            path.write_text(new, encoding='utf-8')
            print('Patched:', path, flush=True)


def build_ops(python, ops, env, log_path):
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            [str(python), 'setup.py', 'build_ext', '--force', 'install'],
            cwd=ops, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors='replace',
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'CountGD CUDA build failed (exit {code}). Full log: {log_path}')
