import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from utils.zbank_autoencoder import ZBankAutoencoder, ZBankDataset, train_autoencoder
from model.transformer_ae import AnomalyTransformerAE


def test_autoencoder_training():
    model = AnomalyTransformerAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
    )
    dummy = torch.zeros(1, 4, 1)
    for _ in range(3):
        model(dummy)
    dataset = ZBankDataset(model.z_bank)
    ae = ZBankAutoencoder(latent_dim=2, enc_in=1, win_size=4)
    train_autoencoder(ae, dataset, epochs=1, batch_size=1)
    out = ae(dataset[0][0].unsqueeze(0))
    assert out.shape == (1, 4, 1)
