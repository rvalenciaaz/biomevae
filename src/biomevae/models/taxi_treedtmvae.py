"""TAXI-TreeDTM-VAE: taxonomy-protected conditional invariance for biomevae.

TAXI = Taxonomy-Anchored eXchangeable Invariance.

Core idea
---------
Existing PhyloDIVA-style models adversarially scrub study identity from the
predictive latent z_y. That can hurt when the disease signal is taxonomic and
the taxonomic signal is itself study-informative.

TAXI instead splits the predictive information into:

    z_tau  := protected taxonomy/class latent     (implemented as DIVA z_y)
    z_rho  := residual predictive latent          (implemented as DIVA z_x)
    z_d    := study/domain reconstruction latent  (implemented as DIVA z_d)

and enforces only the conditional residual invariance:

    z_rho ⊥ D | z_tau, Y

rather than the marginal invariance:

    z_y ⊥ D.

This module subclasses DIVATreeDTMVAE so it can be dropped into the existing
Tree-DTM / DIVA training path with minimal changes.

Recommended trainer call
------------------------
    out = model(node_values, domain=domain, klass=klass)

    loss, metrics = model.loss(
        node_values,
        domain,
        klass=klass,
        out=out,
        beta=args.beta,
        alpha_d=args.alpha_d,
        alpha_y=args.alpha_y,
        lambda_cond_critic=args.lambda_cond_critic,
        lambda_cond_coral=args.lambda_cond_coral,
        lambda_tree_smooth=args.lambda_tree_smooth,
        lambda_orth=args.lambda_orth,
        lambda_tau_aux=args.lambda_tau_aux,
    )

    model.set_grl_lambda(dann_lambda_schedule(progress, args.grl_lambda))

Notes
-----
1. z_tau is NEVER passed through a GRL.
2. The conditional critic receives z_tau and y_context detached.
3. Only z_rho receives the adversarial reversed gradient.
4. Conditional CORAL is class-stratified using hard labels when available and
   soft class probabilities otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.diva_treedtmvae import DIVATreeDTMVAE
from biomevae.models.grl import GradientReversal
from biomevae.models.tree_dtm_vae import LikelihoodName, TreeTopology


__all__ = [
    "ConditionalStudyCritic",
    "TAXIDIVATreeDTMVAE",
    "TAXILossTerms",
    "build_edge_parent_pairs",
    "conditional_coral_by_class_domain",
    "conditional_orthogonality_loss",
    "make_class_context",
    "tree_contrast_smoothness",
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _valid_domain_mask(domain: torch.Tensor, n_domains: int) -> torch.Tensor:
    domain = domain.long()
    return (domain >= 0) & (domain < int(n_domains))


def _valid_class_mask(klass: torch.Tensor, n_classes: int) -> torch.Tensor:
    klass = klass.long()
    return (klass >= 0) & (klass < int(n_classes))


def make_class_context(
    class_logits: torch.Tensor,
    klass: Optional[torch.Tensor],
    *,
    n_classes: int,
) -> torch.Tensor:
    """Return a detached class context for conditional adaptation.

    Labelled rows use the true one-hot class. Unlabelled rows use the model's
    current soft class probabilities.

    Parameters
    ----------
    class_logits:
        Tensor of shape ``(B, C)``.
    klass:
        Tensor of shape ``(B,)`` with labels in ``0..C-1``. Missing labels may
        be encoded as negative values. If ``None``, all rows use soft
        probabilities.
    n_classes:
        Number of phenotype classes.

    Returns
    -------
    context:
        Tensor of shape ``(B, C)`` detached from the graph.
    """
    context = F.softmax(class_logits.detach(), dim=-1)

    if klass is None:
        return context.detach()

    klass = klass.to(class_logits.device).long()
    valid = _valid_class_mask(klass, n_classes)

    if valid.any():
        context = context.clone()
        context[valid] = F.one_hot(
            klass[valid],
            num_classes=int(n_classes),
        ).to(context.dtype)

    return context.detach()


def _weighted_mean_cov(
    z: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_weight: float = 2.0,
    eps: float = 1e-6,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Weighted mean and unbiased-ish covariance.

    The denominator uses the standard Kish-style effective sample correction:

        denom = sum(w) - sum(w^2) / sum(w)

    which reduces to ``n - 1`` for uniform weights.

    Returns ``None`` if the stratum has insufficient effective mass.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be 2D, got shape={tuple(z.shape)}")

    w = weights.to(device=z.device, dtype=z.dtype).clamp_min(0.0)
    mass = w.sum()

    if float(mass.detach().cpu()) < float(min_weight):
        return None

    mean = (w[:, None] * z).sum(dim=0) / mass.clamp_min(eps)
    centered = z - mean

    denom = mass - w.pow(2).sum() / mass.clamp_min(eps)
    if float(denom.detach().cpu()) < eps:
        return None

    cov = (centered * w[:, None]).T @ centered / denom.clamp_min(eps)
    return mean, cov, mass


def _weighted_cross_cov(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_weight: float = 2.0,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """Weighted cross-covariance between ``x`` and ``y``."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same batch size.")

    w = weights.to(device=x.device, dtype=x.dtype).clamp_min(0.0)
    mass = w.sum()

    if float(mass.detach().cpu()) < float(min_weight):
        return None

    x_mean = (w[:, None] * x).sum(dim=0) / mass.clamp_min(eps)
    y_mean = (w[:, None] * y).sum(dim=0) / mass.clamp_min(eps)

    xc = x - x_mean
    yc = y - y_mean

    denom = mass - w.pow(2).sum() / mass.clamp_min(eps)
    if float(denom.detach().cpu()) < eps:
        return None

    return (xc * w[:, None]).T @ yc / denom.clamp_min(eps)


# ---------------------------------------------------------------------------
# Conditional residual invariance losses
# ---------------------------------------------------------------------------


class ConditionalStudyCritic(nn.Module):
    """Gradient-reversed conditional study critic.

    The critic predicts study/domain from:

        [ GRL(z_rho), stopgrad(z_tau), stopgrad(y_context) ]

    Therefore the adversarial gradient reaches only ``z_rho``. The protected
    taxonomy/class latent ``z_tau`` is used as conditioning context but is not
    scrubbed.

    Parameters
    ----------
    dim_rho:
        Dimension of residual latent z_rho.
    dim_tau:
        Dimension of protected taxonomy/class latent z_tau.
    n_classes:
        Number of phenotype classes.
    n_domains:
        Number of studies/domains.
    hidden:
        Hidden width.
    dropout:
        Dropout rate.
    grl_lambda:
        Initial GRL strength.
    """

    def __init__(
        self,
        *,
        dim_rho: int,
        dim_tau: int,
        n_classes: int,
        n_domains: int,
        hidden: int = 128,
        dropout: float = 0.1,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        if min(dim_rho, dim_tau, n_classes, n_domains) <= 0:
            raise ValueError("All dimensions and category counts must be positive.")

        self.dim_rho = int(dim_rho)
        self.dim_tau = int(dim_tau)
        self.n_classes = int(n_classes)
        self.n_domains = int(n_domains)

        self.grl = GradientReversal(lambda_=float(grl_lambda))
        self.net = nn.Sequential(
            nn.Linear(self.dim_rho + self.dim_tau + self.n_classes, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.n_domains),
        )

    def set_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(float(lambda_))

    def logits(
        self,
        z_rho: torch.Tensor,
        z_tau: torch.Tensor,
        y_context: torch.Tensor,
    ) -> torch.Tensor:
        if z_rho.ndim != 2 or z_tau.ndim != 2 or y_context.ndim != 2:
            raise ValueError("z_rho, z_tau and y_context must be 2D tensors.")

        if z_rho.shape[0] != z_tau.shape[0] or z_rho.shape[0] != y_context.shape[0]:
            raise ValueError("z_rho, z_tau and y_context must have same batch size.")

        if z_rho.shape[1] != self.dim_rho:
            raise ValueError(f"Expected z_rho dim {self.dim_rho}, got {z_rho.shape[1]}.")

        if z_tau.shape[1] != self.dim_tau:
            raise ValueError(f"Expected z_tau dim {self.dim_tau}, got {z_tau.shape[1]}.")

        if y_context.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected y_context dim {self.n_classes}, got {y_context.shape[1]}."
            )

        inp = torch.cat(
            [
                self.grl(z_rho),
                z_tau.detach(),
                y_context.detach(),
            ],
            dim=-1,
        )
        return self.net(inp)

    def loss(
        self,
        z_rho: torch.Tensor,
        z_tau: torch.Tensor,
        y_context: torch.Tensor,
        domain: torch.Tensor,
    ) -> torch.Tensor:
        domain = domain.to(z_rho.device).long()
        valid = _valid_domain_mask(domain, self.n_domains)

        if not valid.any():
            return z_rho.new_zeros(())

        logits = self.logits(
            z_rho[valid],
            z_tau[valid],
            y_context[valid],
        )
        return F.cross_entropy(logits, domain[valid])


def conditional_coral_by_class_domain(
    z_rho: torch.Tensor,
    domain: torch.Tensor,
    y_context: torch.Tensor,
    *,
    n_domains: int,
    min_weight: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Class-conditional multi-domain CORAL on z_rho.

    For every class c, compute the covariance of z_rho inside each domain using
    weights:

        w_i = 1{domain_i = d} * p_i(Y=c)

    where p_i is one-hot for labelled samples and soft for unlabelled samples.
    Then penalize pairwise Frobenius distances between domain covariances inside
    the same class.

    Returns zero with graph if no class/domain pair has enough mass.
    """
    if z_rho.ndim != 2:
        raise ValueError("z_rho must be 2D.")
    if y_context.ndim != 2:
        raise ValueError("y_context must be 2D.")

    device = z_rho.device
    dtype = z_rho.dtype
    domain = domain.to(device).long()
    y_context = y_context.to(device=device, dtype=dtype)

    n_classes = int(y_context.shape[1])
    dim = max(1, int(z_rho.shape[1]))

    losses = []

    for c in range(n_classes):
        covs = []

        for d in range(int(n_domains)):
            weights = (domain == d).to(dtype) * y_context[:, c]
            out = _weighted_mean_cov(
                z_rho,
                weights,
                min_weight=min_weight,
                eps=eps,
            )
            if out is None:
                continue
            _, cov, _ = out
            covs.append(cov)

        if len(covs) < 2:
            continue

        for i in range(len(covs)):
            for j in range(i + 1, len(covs)):
                losses.append((covs[i] - covs[j]).pow(2).sum() / (4.0 * dim * dim))

    if not losses:
        return z_rho.new_zeros(())

    return torch.stack(losses).mean()


def conditional_orthogonality_loss(
    z_tau: torch.Tensor,
    z_rho: torch.Tensor,
    domain: torch.Tensor,
    y_context: torch.Tensor,
    *,
    n_domains: int,
    condition_on_domain: bool = True,
    min_weight: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encourage z_tau and z_rho to carry different information.

    This penalizes the squared Frobenius norm of the weighted cross-covariance
    between the protected taxonomy latent and the residual latent. By default,
    it is computed inside each class/domain stratum.

    Setting ``condition_on_domain=False`` computes class-conditional but
    domain-pooled orthogonality.
    """
    if z_tau.ndim != 2 or z_rho.ndim != 2:
        raise ValueError("z_tau and z_rho must be 2D tensors.")

    if z_tau.shape[0] != z_rho.shape[0]:
        raise ValueError("z_tau and z_rho must have same batch size.")

    device = z_tau.device
    dtype = z_tau.dtype
    domain = domain.to(device).long()
    y_context = y_context.to(device=device, dtype=dtype)

    n_classes = int(y_context.shape[1])
    losses = []

    if condition_on_domain:
        domain_ids = list(range(int(n_domains)))
    else:
        domain_ids = [None]

    for c in range(n_classes):
        for d in domain_ids:
            weights = y_context[:, c]
            if d is not None:
                weights = weights * (domain == d).to(dtype)

            cross = _weighted_cross_cov(
                z_tau,
                z_rho,
                weights,
                min_weight=min_weight,
                eps=eps,
            )
            if cross is None:
                continue

            losses.append(cross.pow(2).mean())

    if not losses:
        return z_tau.new_zeros(())

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Tree contrast smoothness on decoder edge logits
# ---------------------------------------------------------------------------


def build_edge_parent_pairs(topo: TreeTopology) -> np.ndarray:
    """Return adjacent edge pairs ``(parent_edge, child_edge)``.

    For edge e = u -> v, its parent edge is the edge ending at u. Root-child
    edges have no parent edge and are skipped.
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
    if edge_logits.ndim != 2:
        raise ValueError("edge_logits must have shape (B, E).")

    batch, n_edges = edge_logits.shape
    n_groups = int(group_sizes.numel())

    idx = edge_to_group.to(edge_logits.device).long()
    if idx.numel() != n_edges:
        raise ValueError(
            f"edge_to_group length {idx.numel()} does not match n_edges {n_edges}."
        )

    idx_b = idx.unsqueeze(0).expand(batch, n_edges)

    group_sum = edge_logits.new_zeros(batch, n_groups)
    group_sum.scatter_add_(1, idx_b, edge_logits)

    sizes = group_sizes.to(edge_logits.device, edge_logits.dtype).clamp_min(1.0)
    group_mean = group_sum / sizes.unsqueeze(0)

    return edge_logits - group_mean.gather(1, idx_b)


def tree_contrast_smoothness(
    edge_logits: torch.Tensor,
    edge_to_group: torch.Tensor,
    group_sizes: torch.Tensor,
    parent_child_edge_pairs: torch.Tensor,
    edge_length: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Shift-invariant smoothness penalty on adjacent centered edge logits."""
    if parent_child_edge_pairs.numel() == 0:
        return edge_logits.new_zeros(())

    centered = _center_edge_logits_by_sibling_group(
        edge_logits,
        edge_to_group,
        group_sizes,
    )

    pairs = parent_child_edge_pairs.to(edge_logits.device).long()
    parent_edge = pairs[:, 0]
    child_edge = pairs[:, 1]

    diff2 = (
        centered.index_select(1, child_edge)
        -
        centered.index_select(1, parent_edge)
    ).pow(2)

    if edge_length is not None:
        length = (
            edge_length.to(edge_logits.device, edge_logits.dtype)
            .index_select(0, child_edge)
            .clamp_min(eps)
        )
        diff2 = diff2 / length.unsqueeze(0)

    return diff2.mean()


# ---------------------------------------------------------------------------
# TAXI model
# ---------------------------------------------------------------------------


@dataclass
class TAXILossTerms:
    """Detached diagnostics for TAXI extras."""

    cond_critic: torch.Tensor
    cond_coral: torch.Tensor
    tree_smooth: torch.Tensor
    orth: torch.Tensor
    tau_aux: torch.Tensor


class TAXIDIVATreeDTMVAE(DIVATreeDTMVAE):
    """Tree-DTM DIVA model with taxonomy-protected conditional invariance.

    This class interprets the base DIVA factors as:

        z_d   := study/domain reconstruction latent
        z_y   := z_tau, protected taxonomy/class latent
        z_x   := z_rho, residual predictive latent

    The classifier uses ``[z_tau, z_rho]``. The conditional domain critic sees
    ``[GRL(z_rho), stopgrad(z_tau), stopgrad(y_context)]``.
    """

    def __init__(
        self,
        *,
        n_domains: int,
        n_classes: int,
        topo: TreeTopology,
        hidden: int = 256,
        latent_d: int = 4,
        latent_tau: int = 8,
        latent_rho: int = 8,
        encoder_layers: int = 2,
        decoder_hidden: int = 256,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        aux_hidden: int = 64,
        critic_hidden: int = 128,
        encoder_pseudocount: float = 0.5,
        init_concentration: float = 50.0,
        likelihood: LikelihoodName = "dirichlet_tree_multinomial",
        aux_on_sample: bool = False,
        class_prior: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
        edge_length: Optional[np.ndarray] = None,
        condition_orth_on_domain: bool = True,
    ) -> None:
        super().__init__(
            n_domains=n_domains,
            n_classes=n_classes,
            topo=topo,
            hidden=hidden,
            latent_d=latent_d,
            latent_y=latent_tau,
            latent_x=latent_rho,
            encoder_layers=encoder_layers,
            decoder_hidden=decoder_hidden,
            decoder_layers=decoder_layers,
            dropout=dropout,
            aux_hidden=aux_hidden,
            encoder_pseudocount=encoder_pseudocount,
            init_concentration=init_concentration,
            likelihood=likelihood,
            aux_on_sample=aux_on_sample,
            class_prior=class_prior,
        )

        self.latent_tau = int(latent_tau)
        self.latent_rho = int(latent_rho)
        self.condition_orth_on_domain = bool(condition_orth_on_domain)

        # Full prediction head uses protected taxonomy/class signal plus
        # residual predictive signal.
        self.taxi_class_head = nn.Sequential(
            nn.Linear(self.latent_tau + self.latent_rho, aux_hidden),
            nn.LayerNorm(aux_hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(aux_hidden, int(n_classes)),
        )

        # Conditional critic scrubs only z_rho, never z_tau.
        self.conditional_critic = ConditionalStudyCritic(
            dim_rho=self.latent_rho,
            dim_tau=self.latent_tau,
            n_classes=int(n_classes),
            n_domains=int(n_domains),
            hidden=int(critic_hidden),
            dropout=float(dropout),
            grl_lambda=float(grl_lambda),
        )

        pairs = build_edge_parent_pairs(topo)
        self.register_buffer(
            "taxi_parent_child_edge_pairs",
            torch.as_tensor(pairs, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "taxi_edge_to_group",
            torch.as_tensor(topo.edge_to_group, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "taxi_group_sizes",
            torch.as_tensor(
                [e - s for s, e in topo.sibling_ranges],
                dtype=torch.float32,
            ),
            persistent=True,
        )

        if edge_length is None:
            edge_length_arr = np.ones(topo.n_edges, dtype=np.float32)
        else:
            edge_length_arr = np.asarray(edge_length, dtype=np.float32)
            if edge_length_arr.shape != (topo.n_edges,):
                raise ValueError(f"edge_length must have shape ({topo.n_edges},).")
            if np.any(~np.isfinite(edge_length_arr)) or np.any(edge_length_arr <= 0):
                raise ValueError("edge_length values must be finite and positive.")

        self.register_buffer(
            "taxi_edge_length",
            torch.as_tensor(edge_length_arr, dtype=torch.float32),
            persistent=True,
        )

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    def set_grl_lambda(self, value: float) -> None:
        self.conditional_critic.set_lambda(float(value))

    def z_tau(self, out: Dict[str, torch.Tensor], *, use_mean: bool = True) -> torch.Tensor:
        return out["mu_y"] if use_mean else out["z_y"]

    def z_rho(self, out: Dict[str, torch.Tensor], *, use_mean: bool = True) -> torch.Tensor:
        return out["mu_x"] if use_mean else out["z_x"]

    def z_domain(self, out: Dict[str, torch.Tensor], *, use_mean: bool = True) -> torch.Tensor:
        return out["mu_d"] if use_mean else out["z_d"]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        node_values: torch.Tensor,
        domain: Optional[torch.Tensor] = None,
        klass: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(node_values, domain=domain, klass=klass)

        # Base DIVATreeDTMVAE's class_logits come from z_y only.
        # Here that is the protected z_tau-only classifier. Keep it for
        # diagnostics and for the optional tau auxiliary loss.
        out["tau_class_logits"] = out["class_logits"]

        class_latent = torch.cat(
            [
                self.z_tau(out, use_mean=not self.aux_on_sample),
                self.z_rho(out, use_mean=not self.aux_on_sample),
            ],
            dim=-1,
        )

        # Main class prediction uses both z_tau and z_rho.
        out["class_logits"] = self.taxi_class_head(class_latent)

        return out

    @torch.no_grad()
    def predict_class(self, node_values: torch.Tensor) -> torch.Tensor:
        out = self.forward(node_values)
        return out["class_logits"]

    @torch.no_grad()
    def embeddings(
        self,
        node_values: torch.Tensor,
        *,
        use_mean: bool = True,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(node_values)
        return {
            "z_tau": self.z_tau(out, use_mean=use_mean),
            "z_rho": self.z_rho(out, use_mean=use_mean),
            "z_d": self.z_domain(out, use_mean=use_mean),
            "z_pred": torch.cat(
                [
                    self.z_tau(out, use_mean=use_mean),
                    self.z_rho(out, use_mean=use_mean),
                ],
                dim=-1,
            ),
            "z_full": torch.cat(
                [
                    self.z_domain(out, use_mean=use_mean),
                    self.z_tau(out, use_mean=use_mean),
                    self.z_rho(out, use_mean=use_mean),
                ],
                dim=-1,
            ),
        }

    # ------------------------------------------------------------------
    # TAXI extra losses
    # ------------------------------------------------------------------

    def extra_losses(
        self,
        out: Dict[str, torch.Tensor],
        domain: torch.Tensor,
        klass: Optional[torch.Tensor] = None,
        *,
        lambda_cond_critic: float = 0.0,
        lambda_cond_coral: float = 0.0,
        lambda_tree_smooth: float = 0.0,
        lambda_orth: float = 0.0,
        lambda_tau_aux: float = 0.0,
        adversary_on_mean: bool = True,
        coral_on_mean: bool = True,
        orth_on_mean: bool = True,
        min_coral_weight: float = 2.0,
        min_orth_weight: float = 4.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute TAXI-specific extra losses.

        Parameters
        ----------
        lambda_cond_critic:
            Weight for conditional adversarial study critic on z_rho.
        lambda_cond_coral:
            Weight for class-conditional CORAL on z_rho.
        lambda_tree_smooth:
            Weight for centered adjacent edge-logit smoothness.
        lambda_orth:
            Weight for conditional cross-covariance penalty between z_tau and
            z_rho.
        lambda_tau_aux:
            Optional auxiliary CE on z_tau alone, preserving the protected
            taxonomy/class channel.
        adversary_on_mean, coral_on_mean, orth_on_mean:
            Whether each loss uses posterior means or sampled latents.
        """
        device = out["mu_y"].device
        domain = domain.to(device).long()

        z_tau_adv = self.z_tau(out, use_mean=adversary_on_mean)
        z_rho_adv = self.z_rho(out, use_mean=adversary_on_mean)

        z_tau_coral = self.z_tau(out, use_mean=coral_on_mean)
        z_rho_coral = self.z_rho(out, use_mean=coral_on_mean)

        z_tau_orth = self.z_tau(out, use_mean=orth_on_mean)
        z_rho_orth = self.z_rho(out, use_mean=orth_on_mean)

        # Use z_tau-only logits for the adaptation context. This avoids letting
        # the residual z_rho determine the condition that is used to scrub z_rho.
        context_logits = out.get("tau_class_logits", out["class_logits"])
        y_context = make_class_context(
            context_logits,
            klass,
            n_classes=self.n_classes,
        )

        total = torch.zeros((), device=device)
        metrics: Dict[str, torch.Tensor] = {}

        if float(lambda_cond_critic) > 0.0:
            critic = self.conditional_critic.loss(
                z_rho_adv,
                z_tau_adv,
                y_context,
                domain,
            )
            scaled = float(lambda_cond_critic) * critic
            total = total + scaled

            metrics["cond_critic"] = critic.detach()
            metrics["cond_critic_scaled"] = scaled.detach()

        if float(lambda_cond_coral) > 0.0:
            coral = conditional_coral_by_class_domain(
                z_rho_coral,
                domain,
                y_context,
                n_domains=self.n_domains,
                min_weight=float(min_coral_weight),
            )
            scaled = float(lambda_cond_coral) * coral
            total = total + scaled

            metrics["cond_coral"] = coral.detach()
            metrics["cond_coral_scaled"] = scaled.detach()

        if float(lambda_tree_smooth) > 0.0:
            smooth = tree_contrast_smoothness(
                out["edge_logits"],
                self.taxi_edge_to_group,
                self.taxi_group_sizes,
                self.taxi_parent_child_edge_pairs,
                self.taxi_edge_length,
            )
            scaled = float(lambda_tree_smooth) * smooth
            total = total + scaled

            metrics["tree_smooth"] = smooth.detach()
            metrics["tree_smooth_scaled"] = scaled.detach()

        if float(lambda_orth) > 0.0:
            orth = conditional_orthogonality_loss(
                z_tau_orth,
                z_rho_orth,
                domain,
                y_context,
                n_domains=self.n_domains,
                condition_on_domain=self.condition_orth_on_domain,
                min_weight=float(min_orth_weight),
            )
            scaled = float(lambda_orth) * orth
            total = total + scaled

            metrics["orth"] = orth.detach()
            metrics["orth_scaled"] = scaled.detach()

        if float(lambda_tau_aux) > 0.0 and klass is not None:
            klass = klass.to(device).long()
            valid_y = _valid_class_mask(klass, self.n_classes)

            if valid_y.any():
                tau_aux = F.cross_entropy(
                    out["tau_class_logits"][valid_y],
                    klass[valid_y],
                )
            else:
                tau_aux = torch.zeros((), device=device)

            scaled = float(lambda_tau_aux) * tau_aux
            total = total + scaled

            metrics["tau_aux"] = tau_aux.detach()
            metrics["tau_aux_scaled"] = scaled.detach()

        metrics.setdefault("cond_critic", torch.zeros((), device=device))
        metrics.setdefault("cond_coral", torch.zeros((), device=device))
        metrics.setdefault("tree_smooth", torch.zeros((), device=device))
        metrics.setdefault("orth", torch.zeros((), device=device))
        metrics.setdefault("tau_aux", torch.zeros((), device=device))

        return total, metrics

    # ------------------------------------------------------------------
    # Full loss
    # ------------------------------------------------------------------

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
        lambda_cond_critic: float = 0.0,
        lambda_cond_coral: float = 0.0,
        lambda_tree_smooth: float = 0.0,
        lambda_orth: float = 0.0,
        lambda_tau_aux: float = 0.0,
        adversary_on_mean: bool = True,
        coral_on_mean: bool = True,
        orth_on_mean: bool = True,
        min_coral_weight: float = 2.0,
        min_orth_weight: float = 4.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full TAXI objective.

        This calls the base DIVA Tree-DTM loss, then adds TAXI extras.
        The base loss includes reconstruction, KLs, q(d|z_d), and the main
        class CE. In this subclass the main class CE uses [z_tau, z_rho].
        """
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
            lambda_cond_critic=lambda_cond_critic,
            lambda_cond_coral=lambda_cond_coral,
            lambda_tree_smooth=lambda_tree_smooth,
            lambda_orth=lambda_orth,
            lambda_tau_aux=lambda_tau_aux,
            adversary_on_mean=adversary_on_mean,
            coral_on_mean=coral_on_mean,
            orth_on_mean=orth_on_mean,
            min_coral_weight=min_coral_weight,
            min_orth_weight=min_orth_weight,
        )

        total = base_loss + extra

        metrics = dict(metrics)
        metrics["loss"] = total.detach()
        metrics["base_loss"] = base_loss.detach()
        metrics["extra_loss"] = extra.detach()
        metrics.update(extra_metrics)

        return total, metrics
