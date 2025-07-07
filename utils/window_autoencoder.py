import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class WindowDataset(Dataset):
    """Wrap a base dataset of windows so that each item returns ``(x, x)``."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, _ = self.base[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)


class BasicWindowAutoencoder(nn.Module):
    """Simple autoencoder operating on windowed time series."""

    def __init__(self, enc_in: int, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, enc_in),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, L, enc_in]
        b, l, c = x.size()
        z = self.encoder(x.view(b * l, c))
        recon = self.decoder(z).view(b, l, c)
        return recon, z.view(b, l, -1)


def train_window_autoencoder(
    model: BasicWindowAutoencoder,
    dataset: Dataset,
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> None:
    """Train ``model`` on ``dataset``."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x, _ in loader:
            recon, _ = model(x)
            loss = loss_fn(recon, x)
            optim.zero_grad()
            loss.backward()
            optim.step()

