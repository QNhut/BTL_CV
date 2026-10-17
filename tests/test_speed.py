import numpy as np
import pytest

from src.evaluation.speed import measure_speed


def test_speed_metadata_and_memory_units():
    torch = pytest.importorskip('torch')

    class Model:
        network = torch.nn.Linear(2, 1)

        def predict_images(self, images):
            return [None for _ in images]

        def num_parameters(self):
            return 3

    result = measure_speed(Model(), [np.zeros((16, 20, 3), np.uint8)], warmup=2, runs=5)
    assert result['fps'] > 0
    assert result['latency_ms_min'] <= result['latency_ms_p95'] <= result['latency_ms_max']
    assert result['benchmark_runs'] == 5 and result['benchmark_warmup'] == 2
    assert result['benchmark_batch_size'] == 1
    assert result['benchmark_device'] == 'cpu'
    assert result['input_shapes_hw'] == [[16, 20]]
    assert result['model_tensor_mb'] == pytest.approx(12 / 1024**2)
    assert 'gpu_peak_allocated_mb' not in result
    with pytest.raises(ValueError):
        measure_speed(Model(), [np.zeros((16, 20, 3), np.uint8)], runs=0)
