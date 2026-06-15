"""PhyloDIVA wrapper for the tree-structured Dirichlet-tree VAE.

This replaces the old ``phylodiva_treenbvae.py`` wrapper. It keeps the
DIVA/tree likelihood framework from ``diva_treedtmvae.py`` and adds optional
domain-generalization regularizers:

1. Domain-adversarial critic on ``z_y`` via a gradient reversal layer.
2. CORAL covariance matching on ``z_x``.
3. Shift-invariant tree-contrast smoothness on centered decoder edge logits.

The third term is deliberately named "tree-contrast smoothness", not
"Brownian-motion smoothness": unless the topology carries calibrated branch
lengths and a valid trait model, it is only a regularization prior.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.diva_treedtmvae import DIVATreeDTMVAE
from biomevae.models.tree_dtm_vae import LikelihoodName, TreeTopology


__all__ = [
    "GradientReversal",
    "LatentDomainAdversary",
    "PhyloDIVATreeDTMVAE",
    "build_edge_parent_pairs",
    "coral_loss_by_domain",
    "tree_contrast_smoothness",
]


class GradientReversal(torch.autograd.Function):
    """Autograd function for gradient reversal."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, float(lambd))


class LatentDomainAdversary(nn.Module):
    """Domain classifier behind a gradient reversal layer."""

    def __init__(
        self,
        latent_dim: int,
        n_domains: int,
        *,
        hidden: int = 64,
        dropout: float = 0.1,
        grl_lambda: float = 1.0,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_domains = int(n_domains)
        self.context_dim = int(context_dim)
        self.grl_lambda = float(grl_lambda)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )

    def set_lambda(self, value: float) -> None:
        self.grl_lambda = float(value)

    def forward(
        self,
        z: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        reverse: bool = True,
    ) -> torch.Tensor:
        if context is not None:
            z = torch.cat([z, context.to(z.device, z.dtype)], dim=-1)
        elif self.context_dim:
            z = torch.cat([z, z.new_zeros(z.size(0), self.context_dim)], dim=-1)

        if reverse:
            z = grad_reverse(z, self.grl_lambda)

        return self.net(z)

    def loss(
        self,
        z: torch.Tensor,
        domain: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        reverse: bool = True,
    ) -> torch.Tensor:
        domain = domain.to(z.device).long()
        mask = (domain >= 0) & (domain < self.n_domains)

        if not mask.any():
            return z.new_zeros(())

        logits = self.forward(
            z[mask],
            context=None if context is None else context[mask],
            reverse=reverse,
        )
        return F.cross_entropy(logits, domain[mask])


def _covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    denom = max(1, x.size(0) - 1)
    return x.t().matmul(x) / float(denom)


def coral_loss_by_domain(
    z: torch.Tensor,
    domain: torch.Tensor,
    *,
    min_per_domain: int = 2,
) -> torch.Tensor:
    """Deep CORAL-style pairwise covariance matching across domains."""
    domain = domain.to(z.device).long()

    valid_domains = []
    for d in torch.unique(domain):
        if d.item() < 0:
            continue
        mask = domain == d
        if int(mask.sum().item()) >= int(min_per_domain):
            valid_domains.append(d)

    if len(valid_domains) < 2:
        return z.new_zeros(())

    covs = [_covariance(z[domain == d]) for d in valid_domains]
    dim = max(1, z.size(1))

    losses = []
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            losses.append((covs[i] - covs[j]).pow(2).sum() / (4.0 * dim * dim))

    return torch.stack(losses).mean() if losses else z.new_zeros(())


def build_edge_parent_pairs(topo: TreeTopology) -> np.ndarray:
    """Return pairs ``(parent_edge, child_edge)`` for adjacent tree edges.

    For edge ``e = u -> v``, its parent edge is the edge ending at ``u``.
    Root-child edges have no parent edge and are skipped.
    """
    child_to_edge = {int(child): i for i, child in enumerate(topo.edge_child)}
    pairs = []

    for child_edge, parent_node in enumerate(topo.edge_parent):
        parent_node = int(parent_node)
        if parent_node in child_to_edge:
            pairs.append((child_to_edge[parent_node], child_edge))

    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)

    return np.asarray(pairs, dtype=np.int64)


def _center_edge_logits_by_sibling_group(
    edge_logits: torch.Tensor,
    edge_to_group: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    """Remove the softmax gauge by subtracting each sibling group's mean logit."""
    batch, n_edges = edge_logits.shape
    n_groups = int(group_sizes.numel())

    idx = edge_to_group.to(edge_logits.device).long().unsqueeze(0).expand(batch, n_edges)

    group_sum = edge_logits.new_zeros(batch, n_groups)
    group_sum.scatter_add_(1, idx, edge_logits)

    group_mean = group_sum / group_sizes.to(edge_logits.device, edge_logits.dtype).unsqueeze(0).clamp_min(1.0)

    return edge_logits - group_mean.gather(1, idx)


def tree_contrast_smoothness(
    edge_logits: torch.Tensor,
    edge_to_group: torch.Tensor,
    group_sizes: torch.Tensor,
    parent_child_edge_pairs: torch.Tensor,
    edge_length: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shift-invariant smoothness penalty on local tree split contrasts.

    The penalty compares centered local logits on adjacent edges. This is safer
    than smoothing raw softmax logits because raw sibling logits are identifiable
    only up to an additive constant within each sibling group.
    """
    if parent_child_edge_pairs.numel() == 0:
        return edge_logits.new_zeros(())

    centered = _center_edge_logits_by_sibling_group(edge_logits, edge_to_group, group_sizes)

    pairs = parent_child_edge_pairs.to(edge_logits.device).long()
    parent_edge = pairs[:, 0]
    child_edge = pairs[:, 1]

    diff2 = (
        centered.index_select(1, child_edge)
        - centered.index_select(1, parent_edge)
    ).pow(2)

    if edge_length is not None:
        length = edge_length.to(edge_logits.device, edge_logits.dtype).index_select(0, child_edge).clamp_min(1e-6)
        diff2 = diff2 / length.unsqueeze(0)

    return diff2.mean()


class PhyloDIVATreeDTMVAE(DIVATreeDTMVAE):
    """DIVA Tree-DTM VAE with optional phylogeny/domain regularizers.

    The base reconstruction and DIVA terms are inherited from
    :class:`DIVATreeDTMVAE`. Extra losses are intentionally optional and should
    be tuned with leave-study-out validation.
    """

    def __init__(
        self,
        *,
        n_domains: int,
        n_classes: int,
        topo: TreeTopology,
        hidden: int = 256,
        latent_d: int = 4,
        latent_y: int = 8,
        latent_x: int = 8,
        encoder_layers: int = 2,
        decoder_hidden: int = 256,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        aux_hidden: int = 64,
        critic_hidden: int = 64,
        encoder_pseudocount: float = 0.5,
        init_concentration: float = 50.0,
        likelihood: LikelihoodName = "dirichlet_tree_multinomial",
        aux_on_sample: bool = False,
        critic_condition_on_class: bool = True,
        grl_lambda: float = 1.0,
        edge_length: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            n_domains=n_domains,
            n_classes=n_classes,
            topo=topo,
            hidden=hidden,
            latent_d=latent_d,
            latent_y=latent_y,
            latent_x=latent_x,
            encoder_layers=encoder_layers,
            decoder_hidden=decoder_hidden,
            decoder_layers=decoder_layers,
            dropout=dropout,
            aux_hidden=aux_hidden,
            encoder_pseudocount=encoder_pseudocount,
            init_concentration=init_concentration,
            likelihood=likelihood,
            aux_on_sample=aux_on_sample,
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

        pairs = build_edge_parent_pairs(topo)
        self.register_buffer(
            "parent_child_edge_pairs",
            torch.as_tensor(pairs, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "edge_to_group",
            torch.as_tensor(topo.edge_to_group, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "group_sizes",
            torch.as_tensor([e - s for s, e in topo.sibling_ranges], dtype=torch.float32),
            persistent=True,
        )

        if edge_length is None:
            edge_length_arr = np.ones(topo.n_edges, dtype=np.float32)
        else:
            edge_length_arr = np.asarray(edge_length, dtype=np.float32)
            if edge_length_arr.shape != (topo.n_edges,):
                raise ValueError(f"edge_length must have shape ({topo.n_edges},).")
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
        lambda_tree_smooth: float = 0.0,
        adversary_on_mean: bool = True,
        coral_on_mean: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = out["mu_y"].device
        total = torch.zeros((), device=device)
        metrics: Dict[str, torch.Tensor] = {}

        if lambda_critic > 0.0:
            z_y = out["mu_y"] if adversary_on_mean else out["z_y"]
            context = self._critic_class_context(out, klass)

            critic = self.study_critic.loss(
                z_y,
                domain,
                context=context,
                reverse=True,
            )
            scaled = float(lambda_critic) * critic
            total = total + scaled

            metrics["critic"] = critic.detach()
            metrics["critic_scaled"] = scaled.detach()

        if lambda_coral > 0.0:
            z_x = out["mu_x"] if coral_on_mean else out["z_x"]

            coral = coral_loss_by_domain(z_x, domain)
            scaled = float(lambda_coral) * coral
            total = total + scaled

            metrics["coral"] = coral.detach()
            metrics["coral_scaled"] = scaled.detach()

        if lambda_tree_smooth > 0.0:
            smooth = tree_contrast_smoothness(
                out["edge_logits"],
                self.edge_to_group,
                self.group_sizes,
                self.parent_child_edge_pairs,
                self.edge_length,
            )
            scaled = float(lambda_tree_smooth) * smooth
            total = total + scaled

            metrics["tree_smooth"] = smooth.detach()
            metrics["tree_smooth_scaled"] = scaled.detach()

        return total, metrics

    def forward(
        self,
        node_values: torch.Tensor,
        domain: Optional[torch.Tensor] = None,
        klass: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(node_values, domain=domain, klass=klass)

        if domain is not None:
            out["domain"] = domain
        if klass is not None:
            out["klass"] = klass

        return out

    def loss(
        self,
        node_values: torch.Tensor,
        domain: torch.Tensor,
        klass: Optional[torch.Tensor] = None,
        out: Optional[Dict[str, torch.Tensor]] = None,
        *,
        likelihood: Optional[LikelihoodName] = None,
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
        lambda_tree_smooth: float = 0.0,
        adversary_on_mean: bool = True,
        coral_on_mean: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(node_values, domain=domain, klass=klass) if out is None else out

        base_loss, metrics = super().loss(
            node_values,
            domain,
            klass=klass,
            out=out,
            likelihood=likelihood,
            beta=beta,
            alpha_d=alpha_d,
            alpha_y=alpha_y,
            unlabelled_y_prior_weight=unlabelled_y_prior_weight,
            free_bits=free_bits,
            concentration_l2=concentration_l2,
            validate_counts=validate_counts,
            observation_pseudocount=observation_pseudocount,
            use_unlabelled_class_marginal=use_unlabelled_class_marginal,
        )

        extra, extra_metrics = self.extra_losses(
            out,
            domain,
            klass,
            lambda_critic=lambda_critic,
            lambda_coral=lambda_coral,
            lambda_tree_smooth=lambda_tree_smooth,
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
