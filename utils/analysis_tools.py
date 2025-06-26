import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

try:
    import ruptures as rpt
except ImportError:  # ruptures might not be installed
    rpt = None


def plot_z_bank_tsne(model, loader, n_samples=500, save_path="z_bank_tsne.png"):
    """Visualize latent vectors stored in ``z_bank`` with t-SNE.

    Parameters
    ----------
    model : AnomalyTransformerWithVAE
        Trained model that keeps previous latent samples.
    loader : torch.utils.data.DataLoader
        Loader providing original time series windows.
    n_samples : int, optional
        Number of samples drawn from each source, by default 500.
    save_path : str, optional
        Where to save the resulting plot.
    """
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

    combined = np.concatenate([orig_latents, replay_latents], axis=0)
    tsne = TSNE(n_components=2, random_state=0)
    reduced = tsne.fit_transform(combined)

    count_orig = orig_latents.shape[0]
    plt.figure()
    plt.scatter(reduced[:count_orig, 0], reduced[:count_orig, 1], s=10, label="Original")
    plt.scatter(reduced[count_orig:, 0], reduced[count_orig:, 1], s=10, label="Replay", alpha=0.7)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("t-SNE of Latent Vectors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_cpd_detection(series, penalty=20, save_path="cpd_detection.png"):
    """Plot change-point locations predicted by ``ruptures``.

    Parameters
    ----------
    series : np.ndarray
        Sequence with shape ``(time, features)`` or ``(time,)``.
    penalty : int, optional
        Penalty value used by ``ruptures.Pelt``, by default ``20``.
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

    algo = rpt.Pelt(model="l2").fit(data)
    result = algo.predict(pen=penalty)

    plt.figure()
    plt.plot(plot_target, label="series")
    for cp in result[:-1]:
        plt.axvline(cp, color="r", linestyle="--", alpha=0.8)
    plt.xlabel("Time")
    plt.title("Change Point Detection")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
