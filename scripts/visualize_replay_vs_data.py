"""Demonstrate comparing replayed samples with actual data."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

missing = []
for _mod in ["numpy", "torch", "matplotlib"]:
    try:
        globals()[_mod] = __import__(_mod)
    except ImportError:
        missing.append(_mod)
if missing:
    raise SystemExit(
        "Missing required packages: "
        + ", ".join(missing)
        + ". Install them with 'pip install -r requirements-demo.txt'"
    )

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset

from utils.analysis_tools import plot_replay_vs_series
from model.transformer_vae import AnomalyTransformerWithVAE


def create_synthetic_series(n_steps=4000):
    """Return a toy time series with a distribution shift."""
    first = np.random.normal(0.0, 1.0, (n_steps // 2, 1))
    second = np.random.normal(3.0, 1.0, (n_steps - n_steps // 2, 1))
    return np.concatenate([first, second], axis=0)


def main():
    series = create_synthetic_series()
    model = AnomalyTransformerWithVAE(
        win_size=20,
        enc_in=1,
        d_model=8,
        n_heads=1,
        e_layers=1,
        d_ff=8,
        latent_dim=4,
        replay_size=2000,
    )

    tensor_series = torch.tensor(series, dtype=torch.float32)
    windows = [tensor_series[i : i + model.win_size]
               for i in range(len(tensor_series) - model.win_size + 1)]
    data = torch.stack(windows)
    loader = DataLoader(TensorDataset(data, torch.zeros(len(data))), batch_size=16)

    with torch.no_grad():
        for batch, _ in loader:
            model(batch)

    plot_replay_vs_series(
        model,
        series.squeeze(),
        start=0,
        end=4000,
        save_path="replay_vs_actual.png",
    )


if __name__ == "__main__":
    main()
