from src.utils.run_layout import evaluation_reports, organize_training_outputs


def test_training_artifacts_grouped_without_moving_checkpoints(tmp_path):
    (tmp_path / 'weights').mkdir()
    for name in ('results.csv', 'args.yaml', 'BoxPR_curve.png', 'config.yaml', 'pred_vs_gt_test.png'):
        (tmp_path / name).write_text('content')
    (tmp_path / 'weights/last.pt').write_text('checkpoint')
    organize_training_outputs(tmp_path)
    assert (tmp_path / 'training/BoxPR_curve.png').exists()
    assert (tmp_path / 'training/results.csv').exists()
    assert (tmp_path / 'config.yaml').exists()
    assert (tmp_path / 'weights/last.pt').exists()
    assert (tmp_path / 'pred_vs_gt_test.png').exists()


def test_grouped_reports_override_legacy_without_duplicate_rows(tmp_path):
    root = tmp_path / 'yolo'
    (root / 'evaluation').mkdir(parents=True)
    old = root / 'metrics_test.json'
    new = root / 'evaluation/metrics_test.json'
    old.write_text('{}')
    new.write_text('{}')
    (root / 'metrics_test_tile.json').write_text('{}')
    reports = evaluation_reports(tmp_path, 'test')
    assert len(reports) == 2
    assert new in reports and old not in reports


def test_migration_preserves_weights_and_existing_evaluation(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    model = tmp_path / 'yolo'
    (model / 'weights').mkdir(parents=True)
    (model / 'evaluation').mkdir()
    (model / 'weights/last.pt').write_text('checkpoint')
    (model / 'metrics_test.json').write_text('legacy')
    (model / 'evaluation/metrics_test.json').write_text('new')
    (model / 'per_image_test.csv').write_text('rows')
    (model / 'results.csv').write_text('history')
    script = Path(__file__).resolve().parents[1] / 'scripts/organize_runs.py'
    result = subprocess.run([sys.executable, str(script), '--runs', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (model / 'weights/last.pt').read_text() == 'checkpoint'
    assert (model / 'evaluation/metrics_test.json').read_text() == 'new'
    assert (model / 'metrics_test.json').read_text() == 'legacy'
    assert (model / 'evaluation/per_image_test.csv').exists()
    assert (model / 'training/results.csv').exists()
