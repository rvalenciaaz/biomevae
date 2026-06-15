"""Domain-Invariant Variational Autoencoder building blocks (Ilse et al. 2020).

DIVA partitions the VAE latent into three independent Gaussian factors —
``z_d`` (domain-specific), ``z_y`` (class-specific) and ``z_x`` (residual) —
each with its own conditional prior:

    p(z_d | d) = N(mu_d(d), sigma_d(d)^2)     # domain-conditional
    p(z_y | y) = N(mu_y(y), sigma_y(y)^2)     # class-conditional
    p(z_x)     = N(0, I)                       # standard

Two auxiliary classifiers — ``q(d | z_d)`` and ``q(y | z_y)`` — push each
factor to carry the corresponding side-information.  The decoder consumes
the concatenated latent ``z = [z_d ; z_y ; z_x]``.

This module is backbone-agnostic: it provides the encoder heads, the
conditional priors, the auxiliary classifiers, the analytic Gaussian KLs
and a thin ``DIVALoss`` orchestrator that returns every term needed to
build an ELBO.  Per-backbone wrappers (TreeNB, Hyp-PhILR-NB) live in
sibling modules and supply the reconstruction likelihood.

References
----------
Ilse, Tomczak, Louizos, Welling, *DIVA: Domain Invariant Variational
Autoencoders*, MIDL 2020.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gaussian KL helpers (independently-normal; sum over latent dims)
# ---------------------------------------------------------------------------

#the clamping of logvar is a safety measure to prevent NaNs from exploding variances
#then the transformation to per-dim KL is the standard closed-form formula for KL between Gaussians.
#(auto encoding variational Bayes, Kingma & Welling 2014, Appendix B.1)

def gaussian_kl_to_standard_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)) per sample, summed over latent dims.

    ``free_bits`` clamps the per-dim KL to a minimum (Kingma et al. 2016)
    so that the encoder does not silence dimensions purely to win a
    fraction of a nat on the ELBO.
    """
    logvar = logvar.clamp(min=-30.0, max=20.0)
    per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    if free_bits > 0:
        per_dim = torch.clamp(per_dim, min=float(free_bits))
    return per_dim.sum(dim=-1)

# KL between two arbitrary Gaussians, per sample, summed over latent dims.

def gaussian_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    *,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) per sample, summed over dims."""
    logvar_q = logvar_q.clamp(min=-30.0, max=20.0)
    logvar_p = logvar_p.clamp(min=-30.0, max=20.0)
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    diff = mu_q - mu_p
    per_dim = 0.5 * (
        (var_q + diff.pow(2)) / var_p - 1.0 + logvar_p - logvar_q
    )
    if free_bits > 0:
        per_dim = torch.clamp(per_dim, min=float(free_bits))
    return per_dim.sum(dim=-1)


# ---------------------------------------------------------------------------
# Encoder heads — three independent (mu, logvar) projections from a shared
# representation.  Backbones build the shared trunk and pass its features
# (``h``) through this module.
# ---------------------------------------------------------------------------


class DIVAEncoderHeads(nn.Module):
    """Three Gaussian heads (z_d, z_y, z_x) on a shared encoder trunk.

    Parameters
    ----------
    feat_dim:
        Width of the trunk's output features.
    latent_d, latent_y, latent_x:
        Latent dimensionality of each factor.  Pick small for ``z_d``
        (it just needs to identify the study/batch) and modest for
        ``z_y`` / ``z_x``.
    logvar_bias:
        Bias init for every ``logvar`` head.  Negative values bias the
        encoder toward small variance early in training and reduce
        posterior-collapse pressure during KL warmup.
    """

    def __init__(
        self,
        feat_dim: int,
        latent_d: int,
        latent_y: int,
        latent_x: int,
        *,
        logvar_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if min(latent_d, latent_y, latent_x) < 1:
            raise ValueError("DIVA latent dims must all be >= 1.")
        self.latent_d = int(latent_d)
        self.latent_y = int(latent_y)
        self.latent_x = int(latent_x)

        self.mu_d = nn.Linear(feat_dim, latent_d)
        self.lv_d = nn.Linear(feat_dim, latent_d)
        self.mu_y = nn.Linear(feat_dim, latent_y)
        self.lv_y = nn.Linear(feat_dim, latent_y)
        self.mu_x = nn.Linear(feat_dim, latent_x)
        self.lv_x = nn.Linear(feat_dim, latent_x)

        for head in (self.lv_d, self.lv_y, self.lv_x):
            nn.init.constant_(head.bias, float(logvar_bias))

    @property
    def total_latent(self) -> int:
        return self.latent_d + self.latent_y + self.latent_x

    def forward(
        self, h: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Return ``(mu_d, lv_d, mu_y, lv_y, mu_x, lv_x)``."""
        mu_d = self.mu_d(h)
        lv_d = self.lv_d(h).clamp(-10.0, 10.0)
        mu_y = self.mu_y(h)
        lv_y = self.lv_y(h).clamp(-10.0, 10.0)
        mu_x = self.mu_x(h)
        lv_x = self.lv_x(h).clamp(-10.0, 10.0)
        return mu_d, lv_d, mu_y, lv_y, mu_x, lv_x

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard Gaussian reparameterisation (tangent-space sampling)."""
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


# ---------------------------------------------------------------------------
# Conditional priors — small lookup-style modules that map a one-hot
# domain or class label to a Gaussian (mu_p, logvar_p).  Embedding +
# linear instead of two separate embedding tables: this keeps the
# parameter count tiny (n_categories * (latent + latent)) and makes
# weight decay applied to embeddings well-defined.
# ---------------------------------------------------------------------------


class CategoryConditionalPrior(nn.Module):
    """Per-category Gaussian prior ``p(z | c) = N(mu(c), sigma(c)^2)``.

    Implemented as two ``nn.Embedding`` tables so the prior is exactly a
    lookup; no MLP, no extra non-linearities.  This is the standard
    DIVA / scANVI parameterisation.

    ``n_categories`` is fixed at construction time; pass the number of
    domains for the domain-conditional prior and the number of classes
    for the class-conditional prior.  When a given category is not
    present in the current minibatch the corresponding rows are simply
    not gradient-updated.
    """

    def __init__(
        self,
        n_categories: int,
        latent_dim: int,
        *,
        logvar_init: float = 0.0,
    ) -> None:
        super().__init__()
        if n_categories < 1:
            raise ValueError("n_categories must be >= 1")
        self.n_categories = int(n_categories)
        self.latent_dim = int(latent_dim)
        self.mu = nn.Embedding(n_categories, latent_dim)
        self.logvar = nn.Embedding(n_categories, latent_dim)
        nn.init.normal_(self.mu.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.logvar.weight, float(logvar_init))

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = c.long()
        return self.mu(c), self.logvar(c).clamp(-10.0, 10.0)


# ---------------------------------------------------------------------------
# Auxiliary classifiers — small MLPs from a single latent factor to
# the corresponding category logits.  Training pushes z_d / z_y to
# carry domain / class information.
# ---------------------------------------------------------------------------


class AuxClassifier(nn.Module):
    """Small MLP: ``z`` → category logits."""

    def __init__(
        self,
        latent_dim: int,
        n_categories: int,
        *,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_categories),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# DIVA loss orchestrator
# ---------------------------------------------------------------------------


@dataclass
class DIVALossOutputs:
    """Bundle of every loss term computed by :class:`DIVALoss`.

    All scalars are batch-mean unless their name says otherwise.  The
    classifier terms carry the *unweighted* cross-entropy; the caller
    multiplies them by ``alpha_d`` / ``alpha_y`` and sums them into the
    final loss.

    Attributes
    ----------
    kl_d, kl_y, kl_x:
        KL divergences for each latent factor.
    ce_d, ce_y:
        Auxiliary cross-entropies (``q(d|z_d)`` / ``q(y|z_y)``).  When
        a class label is missing the cross-entropy is masked out and
        the value here is the mean over the *labelled* portion of the
        batch; check ``n_y_labelled`` before adding to the loss.
    n_y_labelled:
        Number of samples in the batch that had a non-missing class
        label.  Multiply ``ce_y`` by this count (or check >0) before
        adding.
    """

    kl_d: torch.Tensor
    kl_y: torch.Tensor
    kl_x: torch.Tensor
    ce_d: torch.Tensor
    ce_y: torch.Tensor
    n_y_labelled: int


class DIVALoss(nn.Module):
    """Compose every non-reconstruction term of the DIVA ELBO.

    Wires together:
      * :class:`DIVAEncoderHeads`   — produces the three (mu, logvar) pairs;
      * :class:`CategoryConditionalPrior` (one per side-info type);
      * :class:`AuxClassifier`      (one per side-info type).

    ``forward`` consumes the encoder outputs plus the (sampled) latents
    and returns a :class:`DIVALossOutputs` bundle.  The reconstruction
    likelihood lives in the backbone — DIVA itself is likelihood-agnostic.
    """

    def __init__(
        self,
        n_domains: int,
        n_classes: int,
        latent_d: int,
        latent_y: int,
        latent_x: int,
        *,
        aux_hidden: int = 64,
        aux_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prior_d = CategoryConditionalPrior(n_domains, latent_d)
        self.prior_y = CategoryConditionalPrior(n_classes, latent_y)
        self.aux_d = AuxClassifier(
            latent_d, n_domains, hidden=aux_hidden, dropout=aux_dropout,
        )
        self.aux_y = AuxClassifier(
            latent_y, n_classes, hidden=aux_hidden, dropout=aux_dropout,
        )
        self.n_domains = int(n_domains)
        self.n_classes = int(n_classes)

    def forward(
        self,
        *,
        mu_d: torch.Tensor, lv_d: torch.Tensor, z_d: torch.Tensor,
        mu_y: torch.Tensor, lv_y: torch.Tensor, z_y: torch.Tensor,
        mu_x: torch.Tensor, lv_x: torch.Tensor,
        domain: torch.Tensor,
        klass: Optional[torch.Tensor],
        free_bits: float = 0.0,
    ) -> DIVALossOutputs:
        # ---- KL terms --------------------------------------------------
        # z_d against domain-conditional prior (always observed)
        mu_pd, lv_pd = self.prior_d(domain)
        kl_d = gaussian_kl(mu_d, lv_d, mu_pd, lv_pd, free_bits=free_bits).mean()

        # z_y against class-conditional prior when label is known; otherwise
        # against the marginal prior averaged over classes.  In our LOSO
        # setting class labels are always observed, but we keep the
        # semi-supervised branch for full DIVA correctness.
        if klass is not None and (klass >= 0).any():
            valid = klass >= 0
            if valid.all():
                mu_py, lv_py = self.prior_y(klass)
                kl_y = gaussian_kl(
                    mu_y, lv_y, mu_py, lv_py, free_bits=free_bits,
                ).mean()
            else:
                # Mixed-supervision batch: average the two branches with the
                # observed-vs-unobserved counts as weights.
                mu_y_v, lv_y_v = mu_y[valid], lv_y[valid]
                mu_py, lv_py = self.prior_y(klass[valid])
                kl_y_obs = gaussian_kl(
                    mu_y_v, lv_y_v, mu_py, lv_py, free_bits=free_bits,
                ).sum()
                # Unobserved: KL against marginal-of-prior (uniform mixture).
                u_mu = self.prior_y.mu.weight.mean(dim=0, keepdim=True)
                u_lv = self.prior_y.logvar.weight.mean(dim=0, keepdim=True)
                inv = ~valid
                kl_y_unobs = gaussian_kl(
                    mu_y[inv], lv_y[inv],
                    u_mu.expand_as(mu_y[inv]),
                    u_lv.expand_as(lv_y[inv]),
                    free_bits=free_bits,
                ).sum()
                kl_y = (kl_y_obs + kl_y_unobs) / mu_y.size(0)
        else:
            # No labels at all — fall back to N(0,I).  Behaves like a
            # plain VAE on the z_y factor.
            kl_y = gaussian_kl_to_standard_normal(
                mu_y, lv_y, free_bits=free_bits,
            ).mean()

        kl_x = gaussian_kl_to_standard_normal(
            mu_x, lv_x, free_bits=free_bits,
        ).mean()

        # ---- Auxiliary classifiers ------------------------------------
        ce_d = F.cross_entropy(self.aux_d(z_d), domain.long())

        n_y = 0
        if klass is None:
            ce_y = z_y.new_zeros(())
        else:
            valid = klass >= 0
            n_y = int(valid.sum().item())
            if n_y > 0:
                ce_y = F.cross_entropy(
                    self.aux_y(z_y[valid]), klass[valid].long(),
                )
            else:
                ce_y = z_y.new_zeros(())

        return DIVALossOutputs(
            kl_d=kl_d,
            kl_y=kl_y,
            kl_x=kl_x,
            ce_d=ce_d,
            ce_y=ce_y,
            n_y_labelled=n_y,
        )


__all__ = [
    "AuxClassifier",
    "CategoryConditionalPrior",
    "DIVAEncoderHeads",
    "DIVALoss",
    "DIVALossOutputs",
    "gaussian_kl",
    "gaussian_kl_to_standard_normal",
]
