import pytest

torch = pytest.importorskip("torch")

from model.transformer_vae import AnomalyTransformerWithVAE, train_model_with_replay


def test_store_mu_deterministic_replay():
    model = AnomalyTransformerWithVAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
        store_mu=True,
    )
    dummy = torch.zeros(1, 4, 1)
    model(dummy)
    out1 = model.generate_replay_samples(1, deterministic=True)
    out2 = model.generate_replay_samples(1, deterministic=True)
    assert torch.allclose(out1, out2)


def test_freeze_encoder():
    model = AnomalyTransformerWithVAE(
        win_size=4,
        enc_in=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_ff=4,
        latent_dim=2,
        freeze_after=1,
    )
    dummy = torch.zeros(1, 4, 1)
    model(dummy)
    model(dummy)
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_decoder_types():
    dummy = torch.zeros(1, 4, 1)
    for dec in ["mlp", "rnn", "attention"]:
        model = AnomalyTransformerWithVAE(
            win_size=4,
            enc_in=1,
            d_model=4,
            n_heads=1,
            e_layers=1,
            d_ff=4,
            latent_dim=2,
            decoder_type=dec,
        )
        out, _, _, _ = model(dummy)
        assert out.shape == (1, 4, 1)
