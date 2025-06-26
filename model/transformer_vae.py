import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings

from .AnomalyTransformer import EncoderLayer, Encoder
from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding

from utils.utils import my_kl_loss


try:
    import ruptures as rpt
except ImportError:  # ruptures might not be installed
    rpt = None


def my_kl_loss(p, q):
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)



class AnomalyTransformerWithVAE(nn.Module):
    """Anomaly Transformer augmented with a VAE branch."""

    def __init__(self, win_size, enc_in, d_model=512, n_heads=8, e_layers=3,
                 d_ff=512, dropout=0.0, activation='gelu', latent_dim=16,
                 beta: float = 1.0, replay_size: int = 1000,
                 replay_horizon: int | None = None):

        super().__init__()
        self.win_size = win_size
        self.enc_in = enc_in
        self.beta = beta

        self.replay_size = replay_size
        self.replay_horizon = replay_horizon
        self.current_step = 0


        # Transformer components
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False,
                                         attention_dropout=dropout,
                                         output_attention=True),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # VAE components
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, win_size * enc_in)
        )

        # store tuples of (latent_vector, step) for experience replay
        self.z_bank = []
        self.last_mu = None
        self.last_logvar = None

    def _purge_z_bank(self) -> None:
        """Remove stale latent vectors based on ``replay_horizon`` and size."""
        if self.replay_horizon is not None:
            threshold = self.current_step - self.replay_horizon
            self.z_bank = [item for item in self.z_bank if item[1] > threshold]
        if len(self.z_bank) > self.replay_size:
            self.z_bank = self.z_bank[-self.replay_size:]

    def compute_attention_discrepancy(self, series, prior):
        total = 0.0
        for u in range(len(prior)):
            p = series[u]
            q = prior[u] / torch.sum(prior[u], dim=-1, keepdim=True)
            total += torch.mean(my_kl_loss(p, q.detach()))
            total += torch.mean(my_kl_loss(q.detach(), p))
        return total / len(prior)

    def forward(self, x):
        """Forward pass returning reconstruction and attention info."""
        enc = self.embedding(x)
        enc, series, prior, _ = self.encoder(enc)
        pooled = enc.mean(dim=1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z).view(x.size(0), self.win_size, self.enc_in)

        # advance time step and store latent samples for later replay
        self.current_step += 1
        self.z_bank.extend((vec.detach().cpu(), self.current_step) for vec in z)
        self._purge_z_bank()

        self.last_mu = mu
        self.last_logvar = logvar

        return recon, series, prior, z

    def loss_function(self, recon_x, x):
        mse = F.mse_loss(recon_x, x)
        kl = -0.5 * torch.sum(
            1 + self.last_logvar - self.last_mu.pow(2) - self.last_logvar.exp(),
            dim=1)
        kl = torch.mean(kl)
        return mse + self.beta * kl

    def kl_divergence(self):
        """Return KL divergence for the last forward pass."""
        kl = -0.5 * torch.sum(
            1 + self.last_logvar - self.last_mu.pow(2) - self.last_logvar.exp(),
            dim=1)
        return torch.mean(kl)

    def generate_replay_samples(self, n):
        """Generate reconstructions from stored latent vectors."""
        self._purge_z_bank()
        if len(self.z_bank) == 0:
            return None
        idx = np.random.choice(len(self.z_bank), size=min(n, len(self.z_bank)), replace=False)
        z = torch.stack([self.z_bank[i][0] for i in idx])
        z = z.to(next(self.parameters()).device)
        with torch.no_grad():
            recon = self.decoder(z).view(len(idx), self.win_size, self.enc_in)
        return recon


def detect_drift_with_ruptures(window: np.ndarray, pen: int = 20) -> bool:
    if rpt is None:
        raise ImportError("ruptures is required for drift detection")
    # accept batches in (batch, seq_len, features) form
    if window.ndim == 3:
        window = window.reshape(window.shape[0], -1)
    algo = rpt.Pelt(model="l2").fit(window)
    result = algo.predict(pen=pen)
    return len(result) > 1


def train_model_with_replay(model: AnomalyTransformerWithVAE,
                            optimizer: torch.optim.Optimizer,
                            current_data: torch.Tensor) -> tuple[float, bool]:
    model.train()
    data = current_data
    drift_detected = False
    if rpt is not None:
        try:
            drift = detect_drift_with_ruptures(current_data.detach().cpu().numpy())
        except Exception:
            warnings.warn("Change point detection failed; proceeding without replay")
            drift = False
        if drift:
            drift_detected = True
            replay = model.generate_replay_samples(len(current_data))
            if replay is not None:
                data = torch.cat([current_data, replay], dim=0)
    else:
        warnings.warn("ruptures not installed; CPD updates will not run")
    recon, _, _, _ = model(data)
    loss = model.loss_function(recon, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), drift_detected
