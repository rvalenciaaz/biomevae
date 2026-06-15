"""PhyloDIVA wrapper for the plain β-VAE backbone.

The β-VAE has no taxonomy structure, so the BM-smoothness mechanism
does not apply.  PhyloDIVABetaVAE therefore augments
:class:`DIVABetaVAE` with the two backbone-agnostic DA mechanisms
only:

* :class:`LatentStudyCritic` on ``z_y`` (gradient-reversal study
  critic).
* :func:`coral_per_study` on ``z_x``.

Useful as the non-tree reference upper bound — the gain on the
count-likelihood backbones over this row attributes the contribution
of phylogeny-aware machinery (PhILR, tree-softmax decoder, BM
smoothness on tree edges).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from biomevae.losses import reconstruction_loss
from biomevae.models.diva_betavae import DIVABetaVAE
from biomevae.models.phylo_da import LatentStudyCritic, coral_per_study


__all__ = ["PhyloDIVABetaVAE"]


class PhyloDIVABetaVAE(nn.Module):
    """β-VAE with DIVA factors plus GRL on z_y + CORAL on z_x."""

    def __init__(
        self,
        *,
        input_dim: int,
        n_domains: int,
        n_classes: int,
        hidden: List[int] | None = None,
        latent_d: int = 2,
        latent_y: int = 8,
        latent_x: int = 8,
        dropout: float = 0.0,
        activation: str = "leakyrelu",
        layer_norm: bool = False,
        aux_hidden: int = 64,
        critic_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.diva = DIVABetaVAE(
            input_dim=input_dim,
            n_domains=n_domains,
            n_classes=n_classes,
            hidden=hidden,
            latent_d=latent_d,
            latent_y=latent_y,
            latent_x=latent_x,
            dropout=dropout,
            activation=activation,
            layer_norm=layer_norm,
            aux_hidden=aux_hidden,
        )
        self.critic = LatentStudyCritic(
            latent_dim=int(latent_y),
            n_domains=int(n_domains),
            hidden=int(critic_hidden),
            dropout=dropout,
        )

        self.input_dim = input_dim
        self.latent_d = self.diva.latent_d
        self.latent_y = self.diva.latent_y
        self.latent_x = self.diva.latent_x
        self.n_domains = self.diva.n_domains
        self.n_classes = self.diva.n_classes

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.diva.encode(x)

    def reconstruct(self, *args, **kwargs):
        return self.diva.reconstruct(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        domain: torch.Tensor,
        klass: Optional[torch.Tensor],
        *,
        free_bits: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        out = self.diva(x, domain, klass, free_bits=free_bits)
        out["domain"] = domain
        return out

    @staticmethod
    def diva_loss_combine(*args, **kwargs) -> torch.Tensor:
        return DIVABetaVAE.diva_loss_combine(*args, **kwargs)

    def extra_losses(
        self,
        out: Dict[str, torch.Tensor],
        *,
        lambda_bm: float = 0.0,
        lambda_coral: float,
        lambda_critic: float,
    ) -> Dict[str, torch.Tensor]:
        # ``lambda_bm`` is accepted for signature parity but ignored —
        # the β-VAE has no tree-structured decoder output.
        del lambda_bm
        domain = out["domain"]
        extras: Dict[str, torch.Tensor] = {}
        if lambda_critic > 0:
            ce = self.critic.critic_loss(out["z_y"], domain)
            extras["critic"] = float(lambda_critic) * ce
        if lambda_coral > 0:
            cl = coral_per_study(out["z_x"], domain)
            extras["coral"] = float(lambda_coral) * cl
        return extras

    def loss(
        self,
        x: torch.Tensor,
        domain: torch.Tensor,
        klass: Optional[torch.Tensor] = None,
        *,
        recon_kind: str = "mae",
        huber_delta: float = 1.0,
        beta: float = 1.0,
        alpha_d: float = 0.0,
        alpha_y: float = 10.0,
        free_bits: float = 0.0,
        lambda_critic: float = 0.0,
        lambda_coral: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single-batch loss + metrics, matching the taxi / tree-dtm contract."""
        out = self(x, domain, klass, free_bits=free_bits)
        nll = reconstruction_loss(
            out["x"], out["recon"],
            kind=recon_kind, huber_delta=huber_delta,
            per_feature="sum",
        )
        diva_term = DIVABetaVAE.diva_loss_combine(
            out["diva"],
            beta=beta, alpha_d=alpha_d, alpha_y=alpha_y,
            batch_size=out["mu_y"].size(0),
        )
        extras = self.extra_losses(
            out, lambda_coral=lambda_coral, lambda_critic=lambda_critic,
        )
        total = nll + diva_term
        for v in extras.values():
            total = total + v

        metrics: Dict[str, torch.Tensor] = {
            "loss": total.detach(),
            "reconstruction_nll": nll.detach(),
            "diva": diva_term.detach(),
            "kl_d": out["diva"].kl_d.detach(),
            "kl_y": out["diva"].kl_y.detach(),
            "kl_x": out["diva"].kl_x.detach(),
            "ce_d": out["diva"].ce_d.detach(),
            "ce_y": out["diva"].ce_y.detach(),
        }
        for k, v in extras.items():
            metrics[k] = v.detach()
        return total, metrics
