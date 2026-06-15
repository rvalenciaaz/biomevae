"""Domain-adaptation modules for PhyloDIVA.

Two backbone-agnostic pieces:

* :class:`LatentStudyCritic` — a small MLP behind a gradient-reversal
  layer that predicts study identity from a latent slice (typically
  ``z_y``).  The GRL flips the encoder's gradient sign on backward, so
  the encoder is *pushed* to make that latent slice un-predictive of
  study identity.  This is the missing constraint that lets vanilla
  DIVA's ``z_y`` leak study fingerprints in LOSO (see
  ``results/loso_summary_after.tsv``): DIVA only enforces "predict
  domain from ``z_d``" via a soft cross-entropy, never "scrub study
  from ``z_y``".  Adding GRL on ``z_y`` closes that gap.

* :func:`coral_per_study` — Deep CORAL (Sun & Saenko, ECCV 2016) on a
  latent tensor stratified by study.  Returns the mean Frobenius
  distance between per-study covariance matrices, well-defined when at
  least two studies have ≥2 samples in the batch.

Both consume the per-batch ``domain`` tensor that ``build_diva_dataset``
already exposes; no changes are needed in the existing data pipeline.

Why latent-space, not input-space?  An earlier draft used a
hierarchical critic on internal-node abundances aggregated bottom-up
from the leaf-level input.  That critic was a *no-op on the encoder*:
the aggregator has no learnable parameters, so the GRL gradient
flowed back to the leaf input tensor (a dataloader leaf with no
``requires_grad``) and stopped there.  A latent-space critic fixes
this — the path ``z_y → q(d|z_y)`` runs through the entire encoder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.grl import GradientReversal


__all__ = [
    "LatentStudyCritic",
    "dann_lambda_schedule",
    "coral_per_study",
]


# ---------------------------------------------------------------------------
# Latent-space gradient-reversed study critic
# ---------------------------------------------------------------------------


class LatentStudyCritic(nn.Module):
    """Adversarial study classifier on a latent slice (default ``z_y``).

    Architecture: ``z → GRL(λ) → MLP → softmax(n_domains)``.  The GRL
    is the identity on the forward pass and multiplies the gradient by
    ``-λ`` on the backward pass (Ganin & Lempitsky 2015), so the
    encoder learns to make ``z`` un-discriminative of study.  ``λ`` is
    ramped over training via the standard DANN sigmoid schedule
    (:func:`dann_lambda_schedule`).
    """

    def __init__(
        self,
        latent_dim: int,
        n_domains: int,
        *,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grl = GradientReversal(lambda_=1.0)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )
        self.n_domains = int(n_domains)
        self.latent_dim = int(latent_dim)

    def set_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(lambda_)

    def critic_loss(
        self, z: torch.Tensor, domain: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.head(self.grl(z))
        return F.cross_entropy(logits, domain.long())

    def forward(
        self, z: torch.Tensor, domain: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic_loss(z, domain)


def dann_lambda_schedule(
    epoch_t: float, lambda_max: float, *, gamma: float = 10.0,
) -> float:
    """Ganin & Lempitsky (2015) sigmoid ramp ``λ(t) = λ_max·(2/(1+exp(-γt))−1)``.

    ``epoch_t`` is in ``[0, 1]`` (training progress).  Returns ``0`` at
    ``t=0`` and approaches ``λ_max`` as ``t → 1``.
    """
    t = float(max(0.0, min(1.0, epoch_t)))
    return float(lambda_max) * (2.0 / (1.0 + math.exp(-float(gamma) * t)) - 1.0)


# ---------------------------------------------------------------------------
# Deep CORAL on a latent tensor stratified by study
# ---------------------------------------------------------------------------


def coral_per_study(
    z: torch.Tensor, domain: torch.Tensor, *, min_per_study: int = 2,
) -> torch.Tensor:
    """Mean pair-wise Frobenius distance between per-study covariances.

    Returns ``0`` (with-graph) when fewer than two studies have at least
    ``min_per_study`` samples in the batch.  Implements Deep CORAL
    (Sun & Saenko 2016) as a study-symmetric domain-adaptation loss on
    a latent representation.

    The implementation is fully vectorised: a one-hot membership matrix
    drives per-study means and per-study ``Z^T Z`` accumulators via
    matmul / einsum (deterministic under
    ``torch.use_deterministic_algorithms`` when
    ``CUBLAS_WORKSPACE_CONFIG`` is set, matching the original
    matmul-based covariance), and pairwise Frobenius distances are
    materialised as a single ``(K, K)`` reduction masked to ``i < j``
    over study pairs that both pass ``min_per_study``.  No Python loop
    over studies and no per-study ``.item()`` synchronisation; the only
    host-device sync is the one already incurred by ``torch.unique``.
    """
    domain_long = domain.long()
    unique_studies, inverse = torch.unique(domain_long, return_inverse=True)
    K = int(unique_studies.shape[0])
    p = int(z.shape[-1])

    # One-hot membership (B, K) drives all per-study reductions via
    # dense matmul rather than ``index_add_`` so the loss stays
    # bit-equivalent across runs (``index_add_`` is non-deterministic on
    # CUDA — see :func:`biomevae.utils.seeding.set_global_seed`).
    one_hot = F.one_hot(inverse, num_classes=K).to(z.dtype)
    counts = one_hot.sum(dim=0)
    valid_mask = counts >= float(min_per_study)

    # Per-study means.  Counts are clamped only to keep the division
    # finite for invalid studies; their covariance rows are masked out
    # below before they can affect the result.
    sums = one_hot.t() @ z                              # (K, p)
    means = sums / counts.clamp_min(1.0).unsqueeze(-1)
    centred = z - one_hot @ means                       # (B, p)

    # Per-study Z^T Z accumulated as a weighted sum of per-sample outer
    # products.  ``outer`` is (B, p, p); for typical p<=16 this is far
    # smaller than the activations already on-device.  The einsum
    # resolves to a single matmul (``one_hot.T @ outer.flatten``) and is
    # therefore deterministic.
    outer = centred.unsqueeze(-1) * centred.unsqueeze(-2)
    cov_sums = torch.einsum("bk,bpq->kpq", one_hot, outer)
    divisor = (counts - 1.0).clamp_min(1.0)
    covs = cov_sums / divisor.view(K, 1, 1)
    covs = torch.where(valid_mask.view(K, 1, 1), covs, torch.zeros_like(covs))

    diff = covs.unsqueeze(0) - covs.unsqueeze(1)        # (K, K, p, p)
    frob_sq = diff.pow(2).sum(dim=(-1, -2))             # (K, K)
    pair_mask = valid_mask.unsqueeze(0) & valid_mask.unsqueeze(1)
    upper = torch.triu(
        torch.ones((K, K), dtype=torch.bool, device=z.device), diagonal=1,
    )
    mask = (pair_mask & upper).to(frob_sq.dtype)

    total = (frob_sq * mask).sum() / float(4 * p * p)
    n_valid = valid_mask.to(z.dtype).sum()
    n_pairs = (n_valid * (n_valid - 1.0) / 2.0).clamp_min(1.0)
    return total / n_pairs
