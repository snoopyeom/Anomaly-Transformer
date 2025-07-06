"""Demonstrate training an autoencoder on z_bank latents."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

missing = []
for _mod in ["numpy", "torch", "sklearn", "matplotlib"]:
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

from model.transformer_ae import AnomalyTransformerAE
from utils.zbank_autoencoder import ZBankAutoencoder, ZBankDataset, train_autoencoder
from utils.analysis_tools import (
    plot_reconstruction_tsne,
    plot_reconstruction_pca,
    plot_autoencoder_vs_series,
)


def create_series(n_steps=400):
    first = np.random.normal(0.0, 1.0, (n_steps // 2, 1))
    second = np.random.normal(3.0, 1.0, (n_steps - n_steps // 2, 1))
    return np.concatenate([first, second], axis=0)


def main():
    series = create_series()
    model = AnomalyTransformerAE(win_size=20, enc_in=1, d_model=8, n_heads=1,
                                 e_layers=1, d_ff=8, latent_dim=4, replay_size=200)
    tensor_series = torch.tensor(series, dtype=torch.float32)
    windows = [tensor_series[i:i + model.win_size] for i in range(len(tensor_series) - model.win_size + 1)]
    data = torch.stack(windows)
    loader = DataLoader(TensorDataset(data, torch.zeros(len(data))), batch_size=1)
    with torch.no_grad():
        for batch, _ in loader:
            model(batch)

    dataset = ZBankDataset(model.z_bank)
    ae = ZBankAutoencoder(latent_dim=4, enc_in=1, win_size=20)
    train_autoencoder(ae, dataset, epochs=10, batch_size=16)

    plot_reconstruction_tsne(ae, dataset, save_path="recon_tsne.png")
    plot_reconstruction_pca(ae, dataset, save_path="recon_pca.png")
    # Visualize a portion of the original series against the autoencoder
    # reconstruction to qualitatively assess performance
    plot_autoencoder_vs_series(
        ae,
        dataset,
        series.squeeze(),
        start=0,
        end=200,
        save_path="recon_vs_series.png",
    )


if __name__ == "__main__":
    main()
