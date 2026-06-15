"""PhyloDIVA wrapper for the Hyperbolic PhILR-VAE backbone.

Extends :class:`biomevae.models.diva_hyp_philrvae.DIVAHyperbolicPhILRVAE` with
the same three optional domain-generalisation regularisers as
:class:`biomevae.models.phylodiva_treedtmvae.PhyloDIVATreeDTMVAE`:

1. Gradient-reversed study critic on ``z_y`` (optionally conditioned on the
   class context).
2. Deep CORAL covariance matching on ``z_x``.
3. Shift-invariant "PhILR-coord smoothness" — an L2 penalty between adjacent
   contrasts (each contrast attaches to one internal node; its parent
   contrast is the one attached to the parent node). PhILR coordinates are
   already gauge-fixed (no additive constant per sibling group), so no
   sibling-centering step is needed — unlike the raw edge-logit case in the
   Tree-DTM wrapper. This is a smoothness prior, not a Brownian-motion
   likelihood, unless the user supplies calibrated branch lengths via
   ``edge_length``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.diva_hyp_philrvae import DIVAHyperbolicPhILRVAE
from biomevae.models.philrvae import DataKind, LikelihoodName, TaxonomyGraph
from biomevae.models.phylodiva_treedtmvae import (
    LatentDomainAdversary,
    coral_loss_by_domain,
)


__all__ = [
    "PhyloDIVAHyperbolicPhILRVAE",
    "build_philr_parent_contrast_index",
    "philr_coord_smoothness",
]


def build_philr_parent_contrast_index(basis) -> np.ndarray:
    """For each PhILR contrast ``c``, return the index of its parent contrast.

    Each contrast is attached to an internal node (``basis.contrast_node[c]``).
    Its parent contrast is the contrast attached to the parent of that node.
    A contrast whose internal node has no parent contrast (e.g. the
    root-incident contrast) is marked with -1.
    """
    contrast_node = np.asarray(basis.contrast_node, dtype=np.int64)
    parent = np.asarray(basis.parent, dtype=np.int64)

    node_to_contrast: Dict[int, int] = {}
    for c, node in enumerate(contrast_node.tolist()):
        # If multiple contrasts share an internal node (multifurcating SBP),
        # treat the first occurrence as the canonical contrast at that node.
        node_to_contrast.setdefault(int(node), int(c))

    out = np.full(contrast_node.shape, -1, dtype=np.int64)
    for c, node in enumerate(contrast_node.tolist()):
        p = int(parent[node])
        if p >= 0:
            out[c] = node_to_contrast.get(p, -1)
    return out


def philr_coord_smoothness(
    coords: torch.Tensor,
    parent_contrast_idx: torch.Tensor,
    edge_length: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Shift-invariant smoothness penalty on adjacent PhILR contrasts.

    For each contrast ``c`` with a valid parent contrast ``p(c)``, the
    penalty contribution is ``(coords[:, c] - coords[:, p(c)])^2 /
    edge_length[c]``, averaged over all valid contrasts and the batch.
    """
    if coords.ndim < 2:
        raise ValueError("coords must have at least 2 dimensions (..., n_coords).")

    mask = parent_contrast_idx >= 0
    if not bool(mask.any()):
        return coords.new_zeros(())

    valid = mask.nonzero(as_tuple=False).squeeze(-1).long()
    parent_idx = parent_contrast_idx[valid].long()

    child_vals = coords.index_select(-1, valid)
    parent_vals = coords.index_select(-1, parent_idx)
    diff2 = (child_vals - parent_vals).pow(2)

    if edge_length is not None:
        length = edge_length[valid].clamp_min(eps).to(diff2.dtype)
        diff2 = diff2 / length.unsqueeze(0)

    return diff2.mean()


class PhyloDIVAHyperbolicPhILRVAE(DIVAHyperbolicPhILRVAE):
    """DIVA Hyperbolic PhILR-VAE with optional phylogeny/domain regularisers."""

    def __init__(
        self,
        *,
        n_domains: int,
        n_classes: int,
        taxg: TaxonomyGraph,
        curvature: float = 1.0,
        hidden: Optional[Sequence[int]] = None,
        latent_d: int = 4,
        latent_y: int = 8,
        latent_x: int = 8,
        dropout: float = 0.1,
        aux_hidden: int = 64,
        critic_hidden: int = 64,
        count_pseudocount: float = 0.5,
        relative_pseudocount: float = 1e-6,
        default_likelihood: LikelihoodName = "philr_gaussian",
        init_coord_scale: float = 0.5,
        init_concentration: float = 50.0,
        min_coord_scale: float = 1e-4,
        min_concentration: float = 1e-3,
        sort_children: bool = True,
        check_basis: bool = True,
        aux_on_sample: bool = False,
        class_prior: Optional[torch.Tensor] = None,
        critic_condition_on_class: bool = True,
        grl_lambda: float = 1.0,
        edge_length: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            n_domains=n_domains,
            n_classes=n_classes,
            taxg=taxg,
            curvature=curvature,
            hidden=hidden,
            latent_d=latent_d,
            latent_y=latent_y,
            latent_x=latent_x,
            dropout=dropout,
            aux_hidden=aux_hidden,
            count_pseudocount=count_pseudocount,
            relative_pseudocount=relative_pseudocount,
            default_likelihood=default_likelihood,
            init_coord_scale=init_coord_scale,
            init_concentration=init_concentration,
            min_coord_scale=min_coord_scale,
            min_concentration=min_concentration,
            sort_children=sort_children,
            check_basis=check_basis,
            aux_on_sample=aux_on_sample,
            class_prior=class_prior,
        )

        self.critic_condition_on_class = bool(critic_condition_on_class)
        context_dim = int(n_classes) if self.critic_condition_on_class else 0

        self.study_critic = LatentDomainAdversary(
            latent_dim=int(latent_y),
            n_domains=int(n_domains),
            hidden=int(critic_hidden),
            dropout=dropout,
            grl_lambda=grl_lambda,
            context_dim=context_dim,
        )

        parent_contrast = build_philr_parent_contrast_index(self.basis)
        self.register_buffer(
            "parent_contrast_idx",
            torch.as_tensor(parent_contrast, dtype=torch.long),
            persistent=True,
        )

        if edge_length is None:
            edge_length_arr = np.ones(self.basis.n_coords, dtype=np.float32)
        else:
            edge_length_arr = np.asarray(edge_length, dtype=np.float32)
            if edge_length_arr.shape != (self.basis.n_coords,):
                raise ValueError(
                    f"edge_length must have shape ({self.basis.n_coords},)."
                )
            if np.any(~np.isfinite(edge_length_arr)) or np.any(edge_length_arr <= 0):
                raise ValueError("edge_length values must be positive and finite.")

        self.register_buffer(
            "edge_length",
            torch.as_tensor(edge_length_arr, dtype=torch.float32),
            persistent=True,
        )

    def set_grl_lambda(self, value: float) -> None:
        self.study_critic.set_lambda(value)

    def _critic_class_context(
        self,
        out: Dict[str, torch.Tensor],
        klass: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.critic_condition_on_class:
            return None

        logits = out["class_logits"]
        context = F.softmax(logits.detach(), dim=1)

        if klass is not None:
            klass = klass.to(logits.device).long()
            mask = (klass >= 0) & (klass < self.n_classes)
            if mask.any():
                context = context.clone()
                context[mask] = F.one_hot(klass[mask], num_classes=self.n_classes).to(context.dtype)

        return context

    def extra_losses(
        self,
        out: Dict[str, torch.Tensor],
        domain: torch.Tensor,
        klass: Optional[torch.Tensor] = None,
        *,
        lambda_critic: float = 0.0,
        lambda_coral: float = 0.0,
        lambda_philr_smooth: float = 0.0,
        adversary_on_mean: bool = True,
        coral_on_mean: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = out["mu_y"].device
        total = torch.zeros((), device=device)
        metrics: Dict[str, torch.Tensor] = {}

        if lambda_critic > 0.0:
            z_y = out["mu_y"] if adversary_on_mean else out["z_y_tan"]
            context = self._critic_class_context(out, klass)
            critic = self.study_critic.loss(z_y, domain, context=context, reverse=True)
            scaled = float(lambda_critic) * critic
            total = total + scaled
            metrics["critic"] = critic.detach()
            metrics["critic_scaled"] = scaled.detach()

        if lambda_coral > 0.0:
            z_x = out["mu_x"] if coral_on_mean else out["z_x_tan"]
            coral = coral_loss_by_domain(z_x, domain)
            scaled = float(lambda_coral) * coral
            total = total + scaled
            metrics["coral"] = coral.detach()
            metrics["coral_scaled"] = scaled.detach()

        if lambda_philr_smooth > 0.0:
            smooth = philr_coord_smoothness(
                out["coord_mu"],
                self.parent_contrast_idx,
                self.edge_length,
            )
            scaled = float(lambda_philr_smooth) * smooth
            total = total + scaled
            metrics["philr_smooth"] = smooth.detach()
            metrics["philr_smooth_scaled"] = scaled.detach()

        return total, metrics

    def loss(
        self,
        x: torch.Tensor,
        domain: torch.Tensor,
        klass: Optional[torch.Tensor] = None,
        out: Optional[Dict[str, torch.Tensor]] = None,
        *,
        likelihood: Optional[LikelihoodName] = None,
        data_kind: DataKind = "relative",
        beta: float = 1.0,
        alpha_d: float = 1.0,
        alpha_y: float = 1.0,
        unlabelled_y_prior_weight: float = 1.0,
        free_bits: float = 0.0,
        concentration_l2: float = 1e-4,
        validate_counts: bool = True,
        observation_pseudocount: float = 1e-6,
        use_unlabelled_class_marginal: bool = True,
        lambda_critic: float = 0.0,
        lambda_coral: float = 0.0,
        lambda_philr_smooth: float = 0.0,
        adversary_on_mean: bool = True,
        coral_on_mean: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = (
            self.forward(x, domain=domain, klass=klass, data_kind=data_kind)
            if out is None else out
        )

        base_loss, metrics = super().loss(
            x, domain, klass=klass, out=out,
            likelihood=likelihood,
            data_kind=data_kind,
            beta=beta,
            alpha_d=alpha_d, alpha_y=alpha_y,
            unlabelled_y_prior_weight=unlabelled_y_prior_weight,
            free_bits=free_bits,
            concentration_l2=concentration_l2,
            validate_counts=validate_counts,
            observation_pseudocount=observation_pseudocount,
            use_unlabelled_class_marginal=use_unlabelled_class_marginal,
        )

        extra, extra_metrics = self.extra_losses(
            out, domain, klass,
            lambda_critic=lambda_critic,
            lambda_coral=lambda_coral,
            lambda_philr_smooth=lambda_philr_smooth,
            adversary_on_mean=adversary_on_mean,
            coral_on_mean=coral_on_mean,
        )

        total = base_loss + extra

        metrics = dict(metrics)
        metrics["loss"] = total.detach()
        metrics["base_loss"] = base_loss.detach()
        metrics["extra_loss"] = extra.detach()
        metrics.update(extra_metrics)

        return total, metrics
