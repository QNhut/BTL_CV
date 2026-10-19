import csv
import json
import subprocess
import sys
from pathlib import Path


def test_comparison_includes_all_metrics_and_density_missing_values(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runs = tmp_path / 'runs'
    counting = dict(n_images=2, mae=1.0, rmse=1.4, mape_pct=10.0, bias=-1.0,
                    total_rel_error=-0.1, r2=0.9, acc_exact=0.5,
                    acc_within_1=1.0, acc_within_3=1.0)
    detector = dict(model='yolo', family='detection', counting=counting,
                    detection=dict(map50=0.9, map75=0.8, map=0.7, best_f1=0.85,
                                   precision_at_best_f1=0.9, recall_at_best_f1=0.8,
                                   best_f1_score_thr=0.4),
                    speed=dict(fps=20.0, latency_ms_mean=50.0, latency_ms_p95=55.0,
                               params_million=10.0), error_by_count_bin=[])
    density = dict(model='csrnet', family='density', counting={**counting, 'mae': 0.5, 'r2': float('nan')},
                   error_by_count_bin=[])
    for name, report in (('yolo', detector), ('csrnet', density)):
        folder = runs / name
        (folder / 'evaluation').mkdir(parents=True)
        (folder / 'evaluation/metrics_test.json').write_text(json.dumps(report), encoding='utf-8')
    (runs / 'yolo/metrics_test_tile.json').write_text(json.dumps({**detector, 'tiling': True}), encoding='utf-8')
    result = subprocess.run([sys.executable, str(root / 'scripts/compare.py'), '--runs', str(runs)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = runs / 'comparison'
    with (output / 'comparison_test.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and rows[0]['model'] == 'csrnet'
    assert rows[0]['precision_at_best_f1'] == ''
    assert rows[1]['acc_exact'] == '0.500'
    assert rows[1]['map75'] == '0.800'
    assert rows[1]['latency_ms_p95'] == '55.000'
    assert rows[1]['params_million'] == '10.000'
    assert not {'n_images', 'benchmark_device', 'benchmark_hardware', 'benchmark_batch_size',
                'benchmark_runs', 'benchmark_warmup', 'torch_version', 'parameter_dtype',
                'cpu_threads', 'dataset_batch_size'}.intersection(rows[1])
    assert rows[2]['model'] == 'yolo (tile)'
    markdown = (output / 'comparison_test.md').read_text(encoding='utf-8')
    assert '—' in markdown and 'IoU 0.5' in markdown
    assert (output / 'comparison_test.png').exists()
    assert (output / 'deployment_test.csv').exists()
    assert (output / 'deployment_test.md').exists()
    assert (output / 'deployment_test.png').exists()
