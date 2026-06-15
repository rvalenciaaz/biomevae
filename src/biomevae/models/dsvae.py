"""DS-VAE: Disease-Supervised Phylogenetic VAE.

Shared backbone: PhILR transform (Aitchison-isometric, tree-aligned) +
deep MLP encoder + mirror MLP decoder + Negative Binomial likelihood.
Two variants are selected by the boolean ``supervised`` flag:

* ``supervised=False`` — stock β-VAE/PhILR-NB with cyclical β annealing,
  larger latent bottleneck and a free-bits floor.  Targets unsupervised
  embeddings that beat NMF on disease classification.
* ``supervised=True`` — adds a learnable class-conditional Gaussian prior
  ``p(z | y) = N(μ_y, σ²_y)``, a focal-loss classifier head on μ_z, and a
  supervised contrastive loss on ℓ₂-normalised μ_z.  Targets embeddings
  that beat raw-count XGBoost.

The model deliberately does NOT bundle the contrastive / focal / MixUp
losses — those live in :mod:`biomevae.losses` and are composed in the
training loop alongside the NB-NLL and KL terms.  This mirrors the
existing PhILR-VAE / TreeNB-VAE pattern where the model is an encoder /
decoder / prior and the training loop owns the loss schedule.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.tree_spec import TreeSpec
from biomevae.models.philr_treespec import TreeSpecPhILRTransform as PhILRTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orthogonal_frame(n_classes: int, dim: int, scale: float) -> torch.Tensor:
    """Deterministic orthogonal-frame initialiser for class prior means.

    Returns a ``(n_classes, dim)`` matrix whose rows have norm ``scale``
    and are mutually orthogonal (or as close as possible when
    ``n_classes > dim``; in that case the rows become an overcomplete
    frame via QR on a random Gaussian matrix).
    """
    if n_classes <= 0 or dim <= 0:
        return torch.zeros(n_classes, dim)
    # Use a fixed CPU generator so initialisation is deterministic for a
    # given (n_classes, dim) pair.  Downstream seeding in ``train_loop``
    # controls the rest of the model; the class frame should not depend
    # on the caller's seed state because it is effectively a structural
    # prior (think of it like a taxonomy layout).
    gen = torch.Generator().manual_seed(1337)
    g = torch.randn(max(n_classes, dim), dim, generator=gen)
    q, _ = torch.linalg.qr(g)
    frame = q[:n_classes]
    frame = frame / frame.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return frame * float(scale)


# ---------------------------------------------------------------------------
# Class-conditional prior
# ---------------------------------------------------------------------------


class ClassConditionalPrior(nn.Module):
    """Learnable diagonal-Gaussian prior ``p(z|y) = N(μ_y, σ²_y)``.

    ``μ_y`` is initialised on an orthogonal frame scaled to ``sqrt(d)/2``
    so class prototypes start well-separated, and ``log σ²_y`` is
    initialised at ``0`` (unit variance).
    """

    def __init__(self, n_classes: int, latent_dim: int, *, mean_scale: float | None = None):
        super().__init__()
        if n_classes <= 0:
            raise ValueError("n_classes must be positive for ClassConditionalPrior.")
        scale = (float(latent_dim) ** 0.5) / 2.0 if mean_scale is None else float(mean_scale)
        frame = _orthogonal_frame(n_classes, latent_dim, scale)
        self.mu = nn.Parameter(frame.clone())
        self.logvar = nn.Parameter(torch.zeros(n_classes, latent_dim))
        self.n_classes = int(n_classes)
        self.latent_dim = int(latent_dim)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather ``(μ_y, log σ²_y)`` for a batch of integer labels."""
        if y.dtype != torch.long:
            y = y.long()
        mu = self.mu.index_select(0, y)
        logvar = self.logvar.index_select(0, y).clamp(min=-10.0, max=10.0)
        return mu, logvar

    def marginal(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return stacked ``(μ_y, log σ²_y)`` for all classes."""
        return self.mu, self.logvar.clamp(min=-10.0, max=10.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DSVAE(nn.Module):
    """Disease-Supervised Phylogenetic VAE.

    Parameters
    ----------
    n_features:
        Number of input leaves (taxa).  Equal to the length of the PhILR
        contrast matrix's first axis.
    latent_dim:
        Dimensionality of ``μ_z``.  Defaults to 32 (vs PhILR-VAE's 16) to
        relieve the bottleneck on ~1500-feature inputs.
    tree_spec:
        Precomputed tree specification (see
        :func:`biomevae.models.tree_spec.build_tree_spec`).
    supervised:
        When True, attaches a :class:`ClassConditionalPrior` and a
        classifier head.  Training-time losses (focal CE, SupCon, MixUp)
        live in the training loop, not in the model.
    n_classes:
        Required when ``supervised=True``.  Ignored otherwise.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        tree_spec: TreeSpec,
        *,
        supervised: bool = False,
        n_classes: int | None = None,
        hidden: List[int] | None = None,
        dropout: float = 0.1,
        pseudocount: float = 0.5,
        classifier_hidden: int = 128,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = [512, 256, 128]
        self.hidden = list(hidden)
        self.supervised = bool(supervised)
        self.n_features = int(n_features)
        self.latent_dim = int(latent_dim)

        self.philr = PhILRTransform(tree_spec, pseudocount=pseudocount)
        n_coords = self.philr.n_coords  # p - 1

        # --- Encoder ---
        enc: list[nn.Module] = []
        prev = n_coords
        for h in self.hidden:
            enc += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.enc_out_dim = prev  # last hidden size (==128 by default)

        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        nn.init.constant_(self.fc_logvar.bias, -2.0)  # anti-collapse init

        # --- Decoder (mirror) ---
        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(self.hidden):
            dec += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        dec.append(nn.Linear(prev, n_coords))
        self.decoder = nn.Sequential(*dec)

        # --- NB log-dispersion (per-feature, learnable) ---
        self.log_theta = nn.Parameter(torch.full((n_features,), 2.3))

        # --- Supervised heads ---
        self.class_prior: ClassConditionalPrior | None = None
        self.classifier: nn.Sequential | None = None
        self.n_classes: int | None = None
        if self.supervised:
            if n_classes is None or int(n_classes) < 2:
                raise ValueError(
                    "supervised=True requires n_classes >= 2 (got "
                    f"{n_classes!r})."
                )
            self.n_classes = int(n_classes)
            self.class_prior = ClassConditionalPrior(self.n_classes, latent_dim)
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, int(classifier_hidden)),
                nn.LayerNorm(int(classifier_hidden)),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(int(classifier_hidden), self.n_classes),
            )

    # ------------------------------------------------------------------
    # Forward utilities
    # ------------------------------------------------------------------

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = self.philr(x_raw)
        h = self.encoder(coords)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def encode_from_coords(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode directly from PhILR coords (used for MixUp)."""
        h = self.encoder(coords)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def decode(
        self, z: torch.Tensor, library_size: torch.Tensor
    ) -> torch.Tensor:
        coords_hat = self.decoder(z)
        proportions = self.philr.inverse(coords_hat)
        return library_size * proportions

    def classify(self, mu_z: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("classify() requires supervised=True.")
        return self.classifier(mu_z)

    def forward(
        self, x_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mu_x, mu_z, logvar_z)`` — matches PhILR-VAE API."""
        mu_z, logvar_z = self.encode(x_raw)
        z = self.reparam(mu_z, logvar_z)
        library_size = x_raw.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu_x = self.decode(z, library_size)
        return mu_x, mu_z, logvar_z


# ---------------------------------------------------------------------------
# PhILR-space MixUp
# ---------------------------------------------------------------------------


def philr_mixup(
    coords: torch.Tensor,
    y_onehot: torch.Tensor,
    alpha: float = 0.2,
    *,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MixUp on PhILR coordinates + one-hot labels.

    Returns ``(mixed_coords, mixed_labels, lam)`` where ``lam`` is the
    sampled Beta(α, α) coefficient.  When ``alpha <= 0`` this is a no-op
    (λ = 1 and the input is returned unchanged).
    """
    if alpha is None or float(alpha) <= 0.0 or coords.size(0) < 2:
        return coords, y_onehot, coords.new_ones(())
    beta_dist = torch.distributions.Beta(float(alpha), float(alpha))
    lam = beta_dist.sample().to(coords.device)
    if generator is None:
        perm = torch.randperm(coords.size(0), device=coords.device)
    else:
        perm = torch.randperm(coords.size(0), generator=generator).to(coords.device)
    mixed = lam * coords + (1.0 - lam) * coords[perm]
    mixed_y = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
    return mixed, mixed_y, lam
