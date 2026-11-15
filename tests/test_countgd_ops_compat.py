"""Check migration on legacy CUDA source without requiring a GPU."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'countgd_ops_compat', Path(__file__).parents[1] / 'scripts/countgd_ops_compat.py')
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


def test_legacy_cuda_migration_is_repeatable(tmp_path):
    (tmp_path / 'setup.py').touch()
    source = tmp_path / 'src'
    source.mkdir()
    cu = source / 'ops.cu'
    cu.write_text('#include <THC/THCAtomics.cuh>\nvalue.type().is_cuda();\n'
                  'AT_DISPATCH_FLOATING_TYPES(value.type(), "op", f);\nvalue.data<float>();')
    compat.patch_ops(tmp_path)
    migrated = cu.read_text()
    assert 'ATen/cuda/Atomic.cuh' in migrated
    assert 'value.scalar_type()' in migrated
    assert 'value.is_cuda()' in migrated
    assert 'value.data_ptr<float>()' in migrated
    compat.patch_ops(tmp_path)
    assert cu.read_text() == migrated


def test_build_failure_keeps_compiler_diagnostics(tmp_path):
    (tmp_path / 'setup.py').write_text('import sys\nprint("compiler failure", flush=True)\nsys.exit(1)')
    log = tmp_path / 'build.log'
    with pytest.raises(RuntimeError, match='Full log:'):
        compat.build_ops(sys.executable, tmp_path, None, log)
    assert 'compiler failure' in log.read_text()
