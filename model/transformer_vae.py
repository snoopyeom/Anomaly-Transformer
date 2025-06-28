import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
import copy

from .AnomalyTransformer import EncoderLayer, Encoder
from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding

from utils.utils import my_kl_loss, filter_short_segments


try:
    import ruptures as rpt
except ImportError:  # ruptures might not be installed
    rpt = None


def my_kl_loss(p, q):
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class MLPDecoder(nn.Module):
    """Two-layer MLP decoder."""

    def __init__(self, latent_dim: int, d_model: int, win_size: int, enc_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, win_size * enc_in),
        )
        self.win_size = win_size
        self.enc_in = enc_in

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(z.size(0), self.win_size, self.enc_in)


class RNNDecoder(nn.Module):
    """GRU-based decoder for sequential reconstruction."""

    def __init__(self, latent_dim: int, d_model: int, win_size: int, enc_in: int):
        super().__init__()
        self.h_proj = nn.Linear(latent_dim, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.out = nn.Linear(d_model, enc_in)
        self.win_size = win_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h0 = torch.tanh(self.h_proj(z)).unsqueeze(0)
        inputs = torch.zeros(z.size(0), self.win_size, h0.size(-1), device=z.device)
        out, _ = self.rnn(inputs, h0)
        return self.out(out)


class AttentionDecoder(nn.Module):
    """Transformer decoder variant."""

    def __init__(self, latent_dim: int, d_model: int, win_size: int, enc_in: int,
                 n_heads: int, d_ff: int):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.fc = nn.Linear(latent_dim, d_model)
        self.out = nn.Linear(d_model, enc_in)
        self.win_size = win_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        tgt = self.fc(z).unsqueeze(1).repeat(1, self.win_size, 1)
        memory = torch.zeros_like(tgt)
        out = self.decoder(tgt, memory)
        return self.out(out)



class AnomalyTransformerWithVAE(nn.Module):
    """Anomaly Transformer augmented with a VAE branch."""

    def __init__(self, win_size, enc_in, d_model=512, n_heads=8, e_layers=3,
                 d_ff=512, dropout=0.0, activation='gelu', latent_dim=16,
                 beta: float = 1.0, replay_size: int = 1000,
                 replay_horizon: int | None = None,
                 store_mu: bool = False,
                 freeze_after: int | None = None,
                 ema_decay: float | None = None,
                 decoder_type: str = 'mlp'):

        super().__init__()
        self.win_size = win_size
        self.enc_in = enc_in
        self.beta = beta

        self.replay_size = replay_size
        self.replay_horizon = replay_horizon
        self.current_step = 0
        self.store_mu = store_mu
        self.freeze_after = freeze_after
        self.ema_decay = ema_decay
        self.encoder_frozen = False


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
        if decoder_type == 'mlp':
            self.decoder = MLPDecoder(latent_dim, d_model, win_size, enc_in)
        elif decoder_type == 'rnn':
            self.decoder = RNNDecoder(latent_dim, d_model, win_size, enc_in)
        elif decoder_type == 'attention':
            self.decoder = AttentionDecoder(latent_dim, d_model, win_size, enc_in, n_heads, d_ff)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        # store tuples of (latent_vector, step) for experience replay
        self.z_bank = []
        self.last_mu = None
        self.last_logvar = None

        if self.ema_decay is not None:
            self.encoder_ema = copy.deepcopy(self.encoder)
            for p in self.encoder_ema.parameters():
                p.requires_grad_(False)
        else:
            self.encoder_ema = None

    def _purge_z_bank(self) -> None:
        """Remove stale latent vectors based on ``replay_horizon`` and size."""
        if self.replay_horizon is not None:
            threshold = self.current_step - self.replay_horizon
            self.z_bank = [item for item in self.z_bank if item[-1] > threshold]
        if len(self.z_bank) > self.replay_size:
            self.z_bank = self.z_bank[-self.replay_size:]

    def update_ema(self) -> None:
        """Update EMA weights for the encoder."""
        if self.ema_decay is None or self.encoder_ema is None:
            return
        with torch.no_grad():
            for p, p_ema in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
                p_ema.mul_(self.ema_decay)
                p_ema.add_(p * (1.0 - self.ema_decay))

    def maybe_freeze_encoder(self) -> None:
        """Freeze encoder parameters after ``freeze_after`` steps."""
        if (self.freeze_after is not None and
                self.current_step >= self.freeze_after and
                not self.encoder_frozen):
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            self.encoder_frozen = True

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
        recon = self.decoder(z)

        # advance time step and store latent samples for later replay
        self.current_step += 1
        if self.store_mu:
            for mu_i, logvar_i in zip(mu, logvar):
                self.z_bank.append((mu_i.detach().cpu(), logvar_i.detach().cpu(), self.current_step))
        else:
            self.z_bank.extend((vec.detach().cpu(), self.current_step) for vec in z)
        self._purge_z_bank()


        self.last_mu = mu
        self.last_logvar = logvar

        self.maybe_freeze_encoder()

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

    def generate_replay_samples(self, n, deterministic: bool = False):
        """Generate reconstructions from stored latent vectors."""
        self._purge_z_bank()
        if len(self.z_bank) == 0:
            return None
        idx = np.random.choice(len(self.z_bank), size=min(n, len(self.z_bank)), replace=False)
        device = next(self.parameters()).device
        if self.store_mu:
            mus = torch.stack([self.z_bank[i][0] for i in idx]).to(device)
            logvars = torch.stack([self.z_bank[i][1] for i in idx]).to(device)
            if deterministic:
                z = mus
            else:
                std = torch.exp(0.5 * logvars)
                eps = torch.randn_like(std)
                z = mus + eps * std
        else:
            z = torch.stack([self.z_bank[i][0] for i in idx]).to(device)
        with torch.no_grad():
            recon = self.decoder(z)
        return recon


def detect_drift_with_ruptures(window: np.ndarray, pen: int = 20, min_gap: int = 30) -> bool:
    if rpt is None:
        raise ImportError("ruptures is required for drift detection")
    # accept batches in (batch, seq_len, features) form
    if window.ndim == 3:
        window = window.reshape(window.shape[0], -1)
    algo = rpt.Pelt(model="l2").fit(window)
    result = algo.predict(pen=pen)
    result = filter_short_segments(result, min_gap)
    return len(result) > 1


def train_model_with_replay(
    model: AnomalyTransformerWithVAE,
    optimizer: torch.optim.Optimizer,
    current_data: torch.Tensor,
    cpd_penalty: int = 20,
    min_gap: int = 30,
) -> tuple[float, bool]:
    """Train model with replay based on detected concept drift."""
    model.train()
    data = current_data
    drift_detected = False
    if rpt is not None:
        try:
            drift = detect_drift_with_ruptures(
                current_data.detach().cpu().numpy(),
                pen=cpd_penalty,
                min_gap=min_gap,
            )
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
    model.update_ema()
    return loss.item(), drift_detected
