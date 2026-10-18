"""CLI processes must work when launched from a notebook's plotting environment."""
import os
from pathlib import Path
import subprocess
import sys


def test_cli_replaces_inherited_inline_backend():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    env = dict(os.environ, MPLBACKEND='module://matplotlib_inline.backend_inline')
    result = subprocess.run(
        [sys.executable, '-c',
         'import _common; import matplotlib; import matplotlib.pyplot as plt; '
         'assert matplotlib.get_backend().lower() == "agg"; '
         'fig = plt.figure(); plt.close(fig)'],
        cwd=scripts, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
