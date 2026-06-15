"""Smoke tests for :class:`biomevae.models.diva_betavae.DIVABetaVAE`.

Verifies the non-taxonomy DIVA backbone (forward shapes, full /
partial / unsupervised branches, backward through the joint loss,
deterministic reconstruction from latent means).  CPU-only, no
geoopt or torch_geometric.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from biomevae.models.diva_betavae import DIVABetaVAE


def _build_model(input_dim=32, n_domains=5, n_classes=2):
    return DIVABetaVAE(
        input_dim=input_dim,
        n_domains=n_domains,
        n_classes=n_classes,
        hidden=[16, 8],
        latent_d=2,
        latent_y=4,
        latent_x=4,
        activation="leakyrelu",
        layer_norm=False,
        dropout=0.0,
    )


def test_diva_betavae_forward_shapes():
    torch.manual_seed(0)
    m = _build_model()
    x = torch.randn(6, 32)
    out = m(
        x,
        torch.randint(0, 5, (6,)),
        torch.randint(0, 2, (6,)),
    )
    assert out["recon"].shape == (6, 32)
    assert out["mu_d"].shape == (6, 2)
    assert out["mu_y"].shape == (6, 4)
    assert out["mu_x"].shape == (6, 4)
    assert out["z"].shape == (6, 10)


def test_diva_betavae_backward_through_joint_loss():
    torch.manual_seed(1)
    m = _build_model()
    x = torch.randn(6, 32)
    out = m(
        x,
        torch.randint(0, 5, (6,)),
        torch.randint(0, 2, (6,)),
        free_bits=0.02,
    )
    recon = F.mse_loss(out["recon"], x)
    diva_term = DIVABetaVAE.diva_loss_combine(
        out["diva"], beta=0.1, alpha_d=1.0, alpha_y=10.0, batch_size=6,
    )
    (recon + diva_term).backward()
    # Every parameter should have a gradient now.
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad on {name}"


def test_diva_betavae_partial_supervision_counts_correctly():
    m = _build_model()
    x = torch.randn(8, 32)
    klass = torch.tensor([0, 1, -1, 0, 1, -1, -1, 0])
    out = m(x, torch.zeros(8, dtype=torch.long), klass)
    assert out["diva"].n_y_labelled == 5


def test_diva_betavae_unsupervised_branch():
    m = _build_model()
    x = torch.randn(4, 32)
    out = m(x, torch.zeros(4, dtype=torch.long), klass=None)
    assert out["diva"].n_y_labelled == 0
    # ce_y is a zero scalar when klass is None.
    assert out["diva"].ce_y.item() == pytest.approx(0.0, abs=1e-6)


def test_diva_betavae_reconstruct_from_means_matches_decoder():
    torch.manual_seed(2)
    m = _build_model()
    x = torch.randn(5, 32)
    enc = m.encode(x)
    r = m.reconstruct(enc["mu_d"], enc["mu_y"], enc["mu_x"])
    expected = m.decoder(torch.cat([enc["mu_d"], enc["mu_y"], enc["mu_x"]], dim=-1))
    assert torch.allclose(r, expected, atol=1e-6)


def test_diva_betavae_latent_split_inverts_concat():
    m = _build_model()
    mu = torch.randn(3, m.latent_d + m.latent_y + m.latent_x)
    parts = m.latent_split(mu)
    cat = torch.cat([parts["z_d"], parts["z_y"], parts["z_x"]], dim=-1)
    assert torch.allclose(cat, mu)
