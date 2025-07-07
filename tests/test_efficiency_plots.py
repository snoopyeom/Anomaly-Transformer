import pytest

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")

from utils.analysis_tools import (
    plot_memory_usage_curve,
    plot_parameter_update_efficiency,
    plot_latency_vs_model_size,
)


def test_plot_memory_usage_curve(tmp_path):
    steps = np.arange(5)
    cont = steps * 0.5
    batch = np.ones_like(steps)
    out = tmp_path / "mem.png"
    plot_memory_usage_curve(steps, cont, batch, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_parameter_update_efficiency(tmp_path):
    params = np.array([1, 2, 3])
    perf = np.array([0.1, 0.5, 0.8])
    out = tmp_path / "param.png"
    plot_parameter_update_efficiency(params, perf, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_latency_vs_model_size(tmp_path):
    sizes = np.array([10, 20, 30])
    lat = np.array([0.1, 0.2, 0.4])
    out = tmp_path / "latency.png"
    plot_latency_vs_model_size(sizes, lat, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0
