"""Đo tốc độ suy luận (latency / FPS) và kích thước model."""
from __future__ import annotations

import time
import platform
from typing import Sequence

import numpy as np


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def measure_speed(model, images: Sequence[np.ndarray], warmup: int = 5, runs: int = 50) -> dict:
    """Đo thời gian predict_images trên từng ảnh (không tính thời gian đọc ảnh)."""
    images = list(images)
    if not images:
        return {}
    if runs < 1 or warmup < 0:
        raise ValueError('runs must be positive and warmup non-negative')
    import torch
    network = getattr(model, 'network', None)
    if network is None:
        network = getattr(getattr(model, 'model', None), 'model', None)
    device = next(network.parameters()).device if network is not None else torch.device('cpu')
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for i in range(warmup):
        model.predict_images([images[i % len(images)]])
    times = []
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    for k in range(runs):
        im = images[k % len(images)]
        synchronize()
        t0 = time.perf_counter()
        model.predict_images([im])
        synchronize()
        times.append(time.perf_counter() - t0)
    mean = float(np.mean(times))
    out = {
        "latency_ms_mean": mean * 1000,
        "latency_ms_p95": float(np.percentile(times, 95)) * 1000,
        "fps": 1.0 / mean if mean > 0 else float("inf"),
        "latency_ms_median": float(np.median(times)) * 1000,
        "latency_ms_std": float(np.std(times)) * 1000,
        "latency_ms_min": float(np.min(times)) * 1000,
        "latency_ms_max": float(np.max(times)) * 1000,
        "benchmark_total_s": float(np.sum(times)),
        "benchmark_runs": runs,
        "benchmark_warmup": warmup,
        "benchmark_batch_size": 1,
        "benchmark_device": str(device),
        "benchmark_hardware": torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or platform.machine(),
        "torch_version": str(torch.__version__),
        "parameter_dtype": str(next(network.parameters()).dtype) if network is not None else 'unknown',
        "cpu_threads": torch.get_num_threads(),
        "input_shapes_hw": [list(im.shape[:2]) for im in images],
    }
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(device)
        out['gpu_peak_allocated_mb'] = peak / 1024**2
        out['gpu_inference_extra_mb'] = max(0, peak - baseline) / 1024**2
    if network is not None:
        networks = [network, *getattr(model, 'auxiliary_networks', [])]
        tensors = {id(t): t for net in networks for t in [*net.parameters(), *net.buffers()]}
        out['model_tensor_mb'] = sum(t.numel() * t.element_size() for t in tensors.values()) / 1024**2
    n = model.num_parameters()
    if n is not None:
        out["params_million"] = n / 1e6
    return out
