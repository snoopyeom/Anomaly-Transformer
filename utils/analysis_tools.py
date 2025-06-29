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
    replay_latents = torch.stack([z for z, _ in model.z_bank]).cpu().numpy()
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


def visualize_cpd_detection(series, penalty=None, min_size=30,
                            save_path="cpd_detection.png", *,
                            zoom_range=None, top_k=None, zoom_margin=50,
                            extra_zoom_ranges=None):
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
    zoom_range : tuple(int, int), optional
        When set, only ``series[start:end]`` is plotted while the x-axis keeps
        the original indices.
    top_k : int, optional
        Create ``top_k`` additional zoomed views of the most significant change
        points. Saved with ``_top{i}`` suffixes next to ``save_path``.
    zoom_margin : int, optional
        Half-window size around a selected change point for the zoomed views,
        defaulting to ``50``.
    extra_zoom_ranges : list of tuple(int, int), optional
        Additional fixed ranges to visualize. Each range is saved next to
        ``save_path`` with a ``_range{i}`` suffix.
    """
    if rpt is None:
        raise ImportError("ruptures is required for CPD visualization")

    series = np.asarray(series)
    orig_series = series
    if zoom_range is not None:
        start, end = zoom_range
        start = max(start, 0)
        end = min(end, len(series))
        series = series[start:end]
        offset = start
    else:
        offset = 0

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
    x_vals = np.arange(offset, offset + len(plot_target))
    plt.plot(x_vals, plot_target, label="series")
    for cp in result[:-1]:
        plt.axvline(cp + offset, color="r", linestyle="--", alpha=0.8)
    plt.xlabel("Time")
    plt.title("Change Point Detection")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    if top_k:
        metrics = []
        for cp in result[:-1]:
            left = max(cp - zoom_margin, 0)
            right = min(cp + zoom_margin, len(data))
            before = data[left:cp]
            after = data[cp:right]
            var_change = abs(np.var(after) - np.var(before))
            metrics.append((var_change, cp))

        metrics.sort(reverse=True)
        base, ext = os.path.splitext(save_path)
        top_cps = [cp for _, cp in metrics[:top_k]]
        for i, cp in enumerate(top_cps, 1):
            global_cp = cp + offset
            start = max(global_cp - zoom_margin, 0)
            end = min(global_cp + zoom_margin, len(orig_series))
            zoom_path = f"{base}_top{i}{ext}"
            visualize_cpd_detection(
                orig_series, penalty=penalty, min_size=min_size,
                save_path=zoom_path, zoom_range=(start, end),
                top_k=None, zoom_margin=zoom_margin)

    if extra_zoom_ranges:
        base, ext = os.path.splitext(save_path)
        for i, (start, end) in enumerate(extra_zoom_ranges, 1):
            zoom_path = f"{base}_range{i}{ext}"
            visualize_cpd_detection(
                orig_series, penalty=penalty, min_size=min_size,
                save_path=zoom_path, zoom_range=(start, end),
                top_k=None, zoom_margin=zoom_margin)


def plot_replay_vs_series(model, series, *, start=0, end=4000,
                          save_path="replay_vs_series.png"):
    """Compare replay-generated samples with the original series.

    Parameters
    ----------
    model : AnomalyTransformerWithVAE
        Model containing a populated ``z_bank``.
    series : array-like
        1D sequence used during training.
    start : int, optional
        Starting index of the slice to plot.
    end : int, optional
        End index of the slice to plot.
    save_path : str, optional
        Location where the figure will be saved.
    """
    if not model.z_bank:
        raise ValueError("z_bank is empty; train the model before calling")

    series = np.asarray(series).squeeze()
    start = max(0, start)
    end = min(len(series), end)

    n_samples = end - start
    replay = model.generate_replay_samples(n_samples)
    if replay is None:
        raise ValueError("Not enough entries in z_bank for replay")
    replay = replay.detach().cpu().numpy()[:, :, 0]

    win_size = replay.shape[1]
    available = replay.shape[0]
    max_len = min(n_samples, available + win_size - 1)
    recon = np.zeros(max_len)
    counts = np.zeros(max_len)
    for i in range(available):
        idx_start = i
        idx_end = i + win_size
        if idx_start >= max_len:
            break
        win = replay[i]
        if idx_end > max_len:
            win = win[: max_len - idx_start]
            idx_end = max_len
        recon[idx_start:idx_end] += win
        counts[idx_start:idx_end] += 1
    counts[counts == 0] = 1
    recon /= counts

    actual = series[start:start + max_len]
    x = np.arange(start, start + len(actual))
    plt.figure()
    plt.plot(x, actual, label="Actual")
    plt.plot(x[:len(recon)], recon, label="Replay", alpha=0.7)
    plt.xlabel("Time")
    plt.title("Replay vs Actual")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
