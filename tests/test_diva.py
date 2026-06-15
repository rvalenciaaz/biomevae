"""Unit tests for the DIVA core building blocks.

Covers:
* Closed-form Gaussian KL helpers (``gaussian_kl``,
  ``gaussian_kl_to_standard_normal``).
* ``DIVAEncoderHeads`` shapes and reparameterisation.
* ``CategoryConditionalPrior`` lookup behaviour.
* ``DIVALoss`` end-to-end on a tiny tensor (KL terms non-negative,
  cross-entropies finite, partial-supervision branch correct).

These run on CPU and finish in <1 s; they do not require geoopt or
torch_geometric.
"""
from __future__ import annotations

import math

import pytest
import torch

from biomevae.models.diva import (
    AuxClassifier,
    CategoryConditionalPrior,
    DIVAEncoderHeads,
    DIVALoss,
    gaussian_kl,
    gaussian_kl_to_standard_normal,
)


def test_gaussian_kl_zero_when_distributions_match():
    mu = torch.zeros(8, 4)
    lv = torch.zeros(8, 4)
    kl = gaussian_kl_to_standard_normal(mu, lv)
    assert torch.allclose(kl, torch.zeros(8), atol=1e-6)


def test_gaussian_kl_against_diagonal_prior_zero_when_matching():
    mu_q = torch.randn(16, 5)
    lv_q = torch.randn(16, 5)
    kl_self = gaussian_kl(mu_q, lv_q, mu_q, lv_q)
    assert torch.allclose(kl_self, torch.zeros(16), atol=1e-6)


def test_gaussian_kl_free_bits_floors_per_dim():
    mu = torch.zeros(4, 3)  # KL per dim is 0 → would fail without free_bits
    lv = torch.zeros(4, 3)
    fb = 0.05
    kl = gaussian_kl_to_standard_normal(mu, lv, free_bits=fb)
    # Each of 3 dims is clamped to fb; total KL == 3 * fb.
    assert torch.allclose(kl, torch.full((4,), 3 * fb), atol=1e-6)


def test_diva_encoder_heads_shapes_and_reparam():
    h = torch.randn(7, 32)
    heads = DIVAEncoderHeads(feat_dim=32, latent_d=4, latent_y=6, latent_x=8)
    assert heads.total_latent == 18
    mu_d, lv_d, mu_y, lv_y, mu_x, lv_x = heads(h)
    assert mu_d.shape == (7, 4) and lv_d.shape == (7, 4)
    assert mu_y.shape == (7, 6) and lv_y.shape == (7, 6)
    assert mu_x.shape == (7, 8) and lv_x.shape == (7, 8)
    z = DIVAEncoderHeads.reparam(mu_d, lv_d)
    assert z.shape == mu_d.shape
    # logvar bias is -2.0 by default → variances much smaller than 1.
    assert lv_d.exp().mean().item() < 1.0


def test_category_conditional_prior_lookup_independence():
    prior = CategoryConditionalPrior(n_categories=5, latent_dim=3)
    c = torch.tensor([0, 1, 2, 3, 4])
    mu, lv = prior(c)
    # Each row should be the embedding of its own category.
    expected_mu = prior.mu(c)
    assert torch.allclose(mu, expected_mu)
    assert lv.shape == (5, 3)


def test_aux_classifier_returns_correct_logit_shape():
    z = torch.randn(11, 8)
    clf = AuxClassifier(latent_dim=8, n_categories=4, hidden=16)
    logits = clf(z)
    assert logits.shape == (11, 4)


def test_diva_loss_full_supervision_finite_and_nonneg_kl():
    torch.manual_seed(0)
    n_dom, n_cls = 6, 2
    latent_d, latent_y, latent_x = 4, 4, 4

    diva = DIVALoss(
        n_domains=n_dom, n_classes=n_cls,
        latent_d=latent_d, latent_y=latent_y, latent_x=latent_x,
    )
    bsz = 12
    mu_d, lv_d, z_d = (
        torch.randn(bsz, latent_d),
        torch.randn(bsz, latent_d) * 0.3,
        torch.randn(bsz, latent_d),
    )
    mu_y, lv_y, z_y = (
        torch.randn(bsz, latent_y),
        torch.randn(bsz, latent_y) * 0.3,
        torch.randn(bsz, latent_y),
    )
    mu_x, lv_x = torch.randn(bsz, latent_x), torch.randn(bsz, latent_x) * 0.3
    domain = torch.randint(0, n_dom, (bsz,))
    klass = torch.randint(0, n_cls, (bsz,))

    out = diva(
        mu_d=mu_d, lv_d=lv_d, z_d=z_d,
        mu_y=mu_y, lv_y=lv_y, z_y=z_y,
        mu_x=mu_x, lv_x=lv_x,
        domain=domain, klass=klass,
    )
    for term in (out.kl_d, out.kl_y, out.kl_x):
        assert torch.isfinite(term)
        # KL is non-negative for diagonal Gaussians (with free_bits=0).
        assert term.item() >= -1e-6
    assert torch.isfinite(out.ce_d)
    assert torch.isfinite(out.ce_y)
    assert out.n_y_labelled == bsz


def test_diva_loss_partial_supervision_uses_observed_only():
    torch.manual_seed(1)
    diva = DIVALoss(
        n_domains=3, n_classes=2, latent_d=2, latent_y=2, latent_x=2,
    )
    bsz = 8
    args = dict(
        mu_d=torch.zeros(bsz, 2), lv_d=torch.zeros(bsz, 2),
        z_d=torch.zeros(bsz, 2),
        mu_y=torch.zeros(bsz, 2), lv_y=torch.zeros(bsz, 2),
        z_y=torch.zeros(bsz, 2),
        mu_x=torch.zeros(bsz, 2), lv_x=torch.zeros(bsz, 2),
        domain=torch.zeros(bsz, dtype=torch.long),
    )
    klass = torch.tensor([0, 1, 0, 1, -1, -1, -1, -1])
    out = diva(klass=klass, **args)
    assert out.n_y_labelled == 4
    # ce_y must be finite, computed only on the labelled four samples.
    assert torch.isfinite(out.ce_y)


def test_diva_loss_no_labels_falls_back_to_unit_normal_for_z_y():
    diva = DIVALoss(
        n_domains=3, n_classes=2, latent_d=2, latent_y=2, latent_x=2,
    )
    bsz = 4
    args = dict(
        mu_d=torch.zeros(bsz, 2), lv_d=torch.zeros(bsz, 2),
        z_d=torch.zeros(bsz, 2),
        mu_y=torch.zeros(bsz, 2), lv_y=torch.zeros(bsz, 2),
        z_y=torch.zeros(bsz, 2),
        mu_x=torch.zeros(bsz, 2), lv_x=torch.zeros(bsz, 2),
        domain=torch.zeros(bsz, dtype=torch.long),
    )
    out = diva(klass=None, **args)
    # KL(N(0, I) || N(0, I)) == 0
    assert out.kl_y.item() == pytest.approx(0.0, abs=1e-6)
    assert out.n_y_labelled == 0
