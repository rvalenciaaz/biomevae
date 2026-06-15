"""Hyperbolic PhILR-VAE on the new compositional PhILR backbone.

The latent space is the Poincaré ball; the encoder produces a tangent-space
Gaussian which is mapped to the ball via the exponential map at the origin,
and the decoder ``logmap0``-projects the ball point back to the tangent
space *before* applying any Euclidean linear layers. This closes audit D2
("``HyperbolicVAE`` decoder receives Poincare ball points via Euclidean
layers") for the PhILR family.

Reconstruction reuses :class:`PhILRVAE`'s likelihood machinery:
``philr_gaussian`` (logistic-normal on the simplex via orthonormal ILR
coordinates), ``multinomial``, ``dirichlet_multinomial``,
``dirichlet_tree_multinomial`` or ``dirichlet_tree``. The NB / ZINB
variants from the previous generation are intentionally removed: NB on
relative-abundance leaves is the wrong density, and ZINB double-models
zeros that the simplex/Dirichlet handles natively.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    import geoopt
except ImportError as exc:  # pragma: no cover - optional dep
    raise ImportError(
        "Hyperbolic PhILR-VAE requires geoopt. "
        "Install with `pip install -e .[hyper]`."
    ) from exc

from biomevae.models.philrvae import (
    DataKind,
    LikelihoodName,
    PhILRVAE,
    TaxonomyGraph,
    _MLP,
)


__all__ = ["HyperbolicPhILRVAE"]


class HyperbolicPhILRVAE(PhILRVAE):
    """PhILR-VAE with a Poincaré-ball latent.

    Encoder lives in tangent space at the origin (Euclidean Gaussian),
    sampled vectors are mapped to the ball with ``expmap0``, and the
    decoder ``logmap0``-projects back before any Linear layers.
    """

    def __init__(
        self,
        taxg: TaxonomyGraph,
        latent_dim: int,
        *,
        curvature: float = 1.0,
        n_features: Optional[int] = None,
        hidden: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
        count_pseudocount: float = 0.5,
        relative_pseudocount: float = 1e-6,
        default_likelihood: LikelihoodName = "philr_gaussian",
        init_coord_scale: float = 0.5,
        init_concentration: float = 50.0,
        min_coord_scale: float = 1e-4,
        min_concentration: float = 1e-3,
        sort_children: bool = True,
        check_basis: bool = True,
    ) -> None:
        if curvature <= 0:
            raise ValueError("curvature must be > 0 (Poincaré ball curvature c).")

        super().__init__(
            taxg,
            latent_dim,
            n_features=n_features,
            hidden=hidden,
            dropout=dropout,
            count_pseudocount=count_pseudocount,
            relative_pseudocount=relative_pseudocount,
            default_likelihood=default_likelihood,
            init_coord_scale=init_coord_scale,
            init_concentration=init_concentration,
            min_coord_scale=min_coord_scale,
            min_concentration=min_concentration,
            sort_children=sort_children,
            check_basis=check_basis,
        )

        self.curvature = float(curvature)
        self.manifold = geoopt.manifolds.PoincareBallExact(c=curvature)

    # ------------------------------------------------------------------
    # Sampling on the Poincaré ball
    # ------------------------------------------------------------------

    def reparam_to_ball(
        self,
        mu_tan: torch.Tensor,
        logvar_tan: torch.Tensor,
    ) -> torch.Tensor:
        """Sample in tangent space and ``expmap0`` to the ball."""
        logvar = logvar_tan.clamp(-30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        v = mu_tan + torch.randn_like(std) * std
        return self.manifold.projx(self.manifold.expmap0(v))

    # ------------------------------------------------------------------
    # Decode: project off the ball before the Linear stack (audit D2)
    # ------------------------------------------------------------------

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_tan = self.manifold.logmap0(z)
        coord_mu = self.decoder(z_tan)
        leaf_prob = self.philr.inverse(coord_mu)
        return {"coord_mu": coord_mu, "leaf_prob": leaf_prob}

    # ------------------------------------------------------------------
    # Forward / reconstruct overrides — return ``z`` as a ball point
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
    ) -> Dict[str, torch.Tensor]:
        obs_coords = self.philr(x, data_kind=data_kind)
        mu_z, logvar_z = self.encode_coords(obs_coords)
        z = self.reparam_to_ball(mu_z, logvar_z)
        dec = self.decode(z)

        return {
            "obs_coords": obs_coords,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "z": z,
            **dec,
        }

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
        use_mean: bool = True,
    ) -> Dict[str, torch.Tensor]:
        obs_coords = self.philr(x, data_kind=data_kind)
        mu_z, logvar_z = self.encode_coords(obs_coords)
        if use_mean:
            z = self.manifold.projx(self.manifold.expmap0(mu_z))
        else:
            z = self.reparam_to_ball(mu_z, logvar_z)
        return self.decode(z)
