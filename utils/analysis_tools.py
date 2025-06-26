import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

try:
    import umap
except ImportError:  # umap-learn might not be installed
    umap = None

try:
    import ruptures as rpt
except ImportError:  # ruptures might not be installed
    rpt = None


def _collect_latents(model, loader, n_samples):
    """Return original and replay latent vectors."""
    device = next(model.parameters()).device
    orig_latents = []
    seen = 0
    for batch, _ in loader:
        batch = batch.to(device).float()
        with torch.no_grad():
            enc = model.embedding(batch)
            enc, _, _, _ = model.encoder(enc)
            pooled = enc.mean(dim=1)
            mu = model.fc_mu(pooled)
        orig_latents.append(mu.cpu())
        seen += len(batch)
        if seen >= n_samples:
            break
    if not orig_latents:
        raise ValueError("loader did not yield any samples")
    orig_latents = torch.cat(orig_latents, dim=0)[:n_samples].numpy()

    if not model.z_bank:
        raise ValueError("z_bank is empty; train the model before calling")
    replay_latents = torch.stack(model.z_bank).cpu().numpy()
    replay_latents = replay_latents[-n_samples:]

    return orig_latents, replay_latents


def _scatter_projection(orig_latents, replay_latents, reduced, title, save_path):
    count_orig = orig_latents.shape[0]
    plt.figure()
    plt.scatter(reduced[:count_orig, 0], reduced[:count_orig, 1], s=10, label="Original")
    plt.scatter(reduced[count_orig:, 0], reduced[count_orig:, 1], s=10, label="Replay", alpha=0.7)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_z_bank_tsne(model, loader, n_samples=500, save_path="z_bank_tsne.png"):
    """Visualize latent vectors stored in ``z_bank`` with t-SNE."""
    orig_latents, replay_latents = _collect_latents(model, loader, n_samples)
    combined = np.concatenate([orig_latents, replay_latents], axis=0)
    reduced = TSNE(n_components=2, random_state=0).fit_transform(combined)
    _scatter_projection(orig_latents, replay_latents, reduced, "t-SNE of Latent Vectors", save_path)


def plot_z_bank_pca(model, loader, n_samples=500, save_path="z_bank_pca.png"):
    """Visualize latent vectors stored in ``z_bank`` with PCA."""
    orig_latents, replay_latents = _collect_latents(model, loader, n_samples)
    combined = np.concatenate([orig_latents, replay_latents], axis=0)
    reduced = PCA(n_components=2).fit_transform(combined)
    _scatter_projection(orig_latents, replay_latents, reduced, "PCA of Latent Vectors", save_path)


def plot_z_bank_umap(model, loader, n_samples=500, save_path="z_bank_umap.png"):
    """Visualize latent vectors stored in ``z_bank`` with UMAP."""
    if umap is None:
        raise ImportError("umap-learn is required for UMAP visualization")
    orig_latents, replay_latents = _collect_latents(model, loader, n_samples)
    combined = np.concatenate([orig_latents, replay_latents], axis=0)
    reducer = umap.UMAP(n_components=2, random_state=0)
    reduced = reducer.fit_transform(combined)
    _scatter_projection(orig_latents, replay_latents, reduced, "UMAP of Latent Vectors", save_path)


def visualize_cpd_detection(series, penalty=None, min_size=30, save_path="cpd_detection.png"):
    """Plot change-point locations predicted by ``ruptures``.

    Parameters
    ----------
    series : np.ndarray
        Sequence with shape ``(time, features)`` or ``(time,)``.
    penalty : float, optional
        Penalty passed to ``rpt.Pelt.predict``. If ``None`` a heuristic based
        on sequence length and variance is used.
    min_size : int, optional
        Minimum distance between change points, defaults to ``30``.
    save_path : str, optional
        Path to save the visualization.
    """
    if rpt is None:
        raise ImportError("ruptures is required for CPD visualization")

    series = np.asarray(series)
    if series.ndim == 1:
        data = series.reshape(-1, 1)
        plot_target = series
    else:
        data = series.reshape(series.shape[0], -1)
        plot_target = series[:, 0]

    if penalty is None:
        penalty = np.log(len(data)) * np.var(data)

    algo = rpt.Pelt(model="l2", min_size=min_size).fit(data)
    result = algo.predict(pen=penalty)

    plt.figure()
    plt.plot(plot_target, label="series")
    for cp in result[:-1]:
        plt.axvline(cp, color="r", linestyle="--", alpha=0.8)
    plt.xlabel("Time")
    plt.title("Change Point Detection")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
