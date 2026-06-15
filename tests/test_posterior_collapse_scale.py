"""Regression tests for the reconstruction-loss / KL scale convention.

Motivation (MetaCardis_2020_a posterior-collapse analysis)
-----------------------------------------------------------
Historically, :func:`biomevae.losses.reconstruction_loss` averaged the
per-element loss over features (``mean(dim=1)``) while
:func:`biomevae.losses.kl_per_sample` summed over latent dims. On the
MetaCardis SGB table (~500 features, latent_dim=16–24) this put the
recon term on a ~1e-2 scale and the KL term on a ~1e1 scale. With the
default ``beta_max=0.05`` the KL contribution ``β·KL ≈ 0.8`` dominated
the recon ``~0.01`` by roughly two orders of magnitude, driving every
VAE that uses the shared ``train_once`` loop (β-VAE, Vanilla, Hyperbolic,
Tax-aware, Hyp+Tax, Graph, TreePrior, PhyloFusion) into posterior
collapse: PC1 variance → 100%, val ELBO diverging, ``active_units → 0``,
and classifier accuracy barely above chance.

NB-NLL based models (TreeNB-VAE, PhILR-VAE, DS-VAE) and Dirichlet-NLL
models (TreeDirichlet-VAE) were unaffected because their likelihood is
already summed over features.

The fix makes ``reconstruction_loss`` sum over features by default
(matching ``nb_nll``/``dirichlet_nll`` and the KL convention). The
``per_feature="mean"`` path is retained for reporting metrics
(``vae_test`` test_report) where a per-feature error is more readable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomevae.losses import (  # noqa: E402
    compute_losses,
    kl_per_sample,
    reconstruction_loss,
)


class TestReconFeatureReduction:
    def test_default_reduction_is_sum(self):
        """Canonical VAE ELBO sums over features."""
        torch.manual_seed(0)
        x = torch.randn(4, 100)
        recon = torch.randn(4, 100)
        r_sum = reconstruction_loss(x, recon, kind="mse")
        r_mean = reconstruction_loss(x, recon, kind="mse", per_feature="mean")
        # The per-feature sum is `input_dim` times the per-feature mean.
        assert torch.isclose(r_sum, r_mean * 100.0, rtol=1e-5)

    def test_mean_reduction_still_available_for_reporting(self):
        """Legacy per-feature-mean scale stays available for test reports."""
        torch.manual_seed(1)
        x = torch.randn(8, 50)
        recon = x + 0.1 * torch.randn_like(x)
        r_mean = reconstruction_loss(x, recon, kind="mae", per_feature="mean")
        # MAE of ~0.1 noise across the batch on per-feature-mean scale.
        assert 0.05 < r_mean.item() < 0.15

    def test_invalid_reduction_raises(self):
        x = torch.zeros(2, 3)
        with pytest.raises(ValueError):
            reconstruction_loss(x, x, kind="mse", per_feature="bogus")


class TestReconKlBalance:
    """Under the new convention a reasonable ``beta_max`` leaves recon ≥ β·KL.

    This is the property that was violated before the fix — the β·KL term
    was ~80× larger than the recon term on MetaCardis-scale inputs, which
    is the arithmetic precondition for posterior collapse under gradient
    descent (the encoder can minimise loss simply by matching the prior).
    """

    def test_recon_dominates_at_prior_with_default_beta(self):
        torch.manual_seed(2)
        # MetaCardis-style shape: hundreds of features, modest latent dim.
        batch, n_feat, n_latent = 16, 500, 16
        x = torch.randn(batch, n_feat)
        # Simulate an encoder pinned at the prior: mu=0, logvar=0 (variance=1).
        mu = torch.zeros(batch, n_latent)
        logvar = torch.zeros(batch, n_latent)
        # Decoder output at zero (worst case for the recon term — this is
        # exactly the state a collapsed VAE ends up in: z ≈ N(0,I), output
        # ≈ decoder bias ≈ dataset mean).
        recon = torch.zeros_like(x)

        loss, r, kl = compute_losses(
            x, recon, mu, logvar,
            recon_kind="mse", objective="beta", beta=0.05, free_bits=0.0,
        )
        # kl_per_sample sums per-dim: for mu=0, logvar=0 the closed-form KL
        # against N(0, I) is zero, so we instead check the magnitudes are
        # comparable when we perturb mu.
        mu = torch.randn(batch, n_latent) * 0.5
        loss, r, kl = compute_losses(
            x, recon, mu, logvar,
            recon_kind="mse", objective="beta", beta=0.05, free_bits=0.0,
        )
        # With the fix, recon is sum-over-features — roughly mean(x**2)*n_feat
        # ≈ 1.0 * 500 = 500. The KL term is 0.05 * ~2 ≈ 0.1. The ratio is
        # now overwhelmingly in favour of recon, which is the correct
        # arithmetic for an encoder to receive a useful gradient.
        assert r.item() > 10.0 * (0.05 * kl.item()), (
            f"β·KL ({0.05 * kl.item():.3f}) should not dominate recon "
            f"({r.item():.3f}) under the canonical ELBO convention."
        )


class TestTrainOnceDoesNotCollapse:
    """End-to-end: ``train_once`` on a correlated toy dataset must keep at
    least one latent unit active when run with the documented defaults for
    the β-VAE family (``beta_max=0.05``, ``kl_warmup`` > 0). Before the
    fix this test would have ended with ``active_units = 0``.
    """

    def _params(self, **overrides):
        base = {
            "objective": "beta",
            "val_split": 0.2,
            "standardize": False,
            "device": "cpu",
            "hidden": [32, 16],
            "latent_dim": 8,
            "dropout": 0.0,
            "activation": "relu",
            "layer_norm": False,
            "optimizer": "adam",
            "lr": 2e-3,
            "weight_decay": 0.0,
            "batch_size": 16,
            "epochs": 20,
            "kl_warmup": 5,
            "beta_max": 0.05,
            "free_bits": 0.0,
            "recon": "mse",
            "huber_delta": 1.0,
            "capacity_start": 0.0,
            "capacity_end": None,
            "capacity_epochs": 10,
            "capacity_gamma": 1.0,
            "grad_clip": 1.0,
            "early_stop": 0,
        }
        base.update(overrides)
        return base

    def test_euclid_keeps_active_units(self, tmp_path):
        from biomevae.trainers.train_loop import train_once

        # Build a toy dataset with real low-dim structure (rank-3 signal +
        # noise over 100 features). A VAE with a working posterior should
        # easily discover 1-3 active units.
        rng = np.random.default_rng(0)
        n, p = 200, 100
        factors = rng.normal(size=(n, 3)).astype(np.float32)
        loadings = rng.normal(size=(3, p)).astype(np.float32)
        X = (factors @ loadings + 0.1 * rng.normal(size=(n, p))).astype(np.float32)

        params = self._params()
        res = train_once(
            X, [f"s{i}" for i in range(n)], str(tmp_path), params,
            seed=1, verbose=False,
        )

        assert res["active_units"] >= 1, (
            "At least one latent unit must survive a default β-VAE run on a "
            "dataset with real low-rank structure — 0 active units is the "
            "posterior-collapse fingerprint this regression test guards "
            "against."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
