from src.utils import save_yaml
from src.utils.report_sessions import discover_models, test_report as find_test_report


def test_discover_uses_configured_run_name_across_sessions(tmp_path):
    configs = tmp_path / 'configs'
    configs.mkdir()
    save_yaml({'name': 'yolov8s'}, configs / 'yolov8.yaml')
    save_yaml({'name': 'csrnet'}, configs / 'csrnet.yaml')
    (tmp_path / 'session1/runs/yolov8s').mkdir(parents=True)
    (tmp_path / 'session2/csrnet').mkdir(parents=True)
    assert discover_models(configs, [tmp_path / 'session1', tmp_path / 'session2']) == ['csrnet', 'yolov8']


def test_metrics_prefer_grouped_base_report_over_legacy_or_tile(tmp_path):
    (tmp_path / 'metrics_test_tile.json').write_text('{}')
    assert find_test_report(tmp_path) is None
    (tmp_path / 'metrics_test.json').write_text('{}')
    (tmp_path / 'evaluation').mkdir()
    grouped = tmp_path / 'evaluation/metrics_test.json'
    grouped.write_text('{}')
    assert find_test_report(tmp_path) == grouped
