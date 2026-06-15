"""TAXI variant of the DIVA Hyperbolic PhILR-VAE.

TAXI = Taxonomy-Anchored eXchangeable Invariance.

This is the PhILR / hyperbolic analogue of
:class:`biomevae.models.taxi_treedtmvae.TAXIDIVATreeDTMVAE`.  Like its
Tree-DTM cousin it reinterprets the DIVA factors as:

    z_d   := study/domain reconstruction latent
    z_y   := z_tau, protected taxonomy/class latent
    z_x   := z_rho, residual predictive latent

and enforces only the conditional residual invariance ``z_rho ⊥ D | z_tau, Y``
instead of DIVA's marginal ``z_y ⊥ D``.  The full prediction head reads
``[z_tau, z_rho]``; the conditional study critic scrubs only ``z_rho``;
``z_tau`` is never passed through a GRL.

Differences vs. :class:`PhyloDIVAHyperbolicPhILRVAE`:

* Marginal CORAL on ``z_x`` is replaced by **class-conditional** CORAL
  on ``z_rho`` (matching covariances inside each phenotype class, so
  cross-study alignment cannot collapse disease-induced taxonomic
  shifts).
* The PhILR-coord smoothness penalty (already shift-invariant) is kept
  as an optional regulariser — identical math to PhyloDIVA's, reused.
* A cross-covariance orthogonality penalty between ``z_tau`` and
  ``z_rho`` (optional) discourages the residual channel from absorbing
  the taxonomy/class signal.
* An optional auxiliary CE on ``z_tau`` alone pins the protected channel
  to the phenotype signal.

The helper functions for the conditional critic, class-conditional
CORAL, conditional orthogonality and the class-context builder are
imported verbatim from :mod:`biomevae.models.taxi_treedtmvae`; the
PhILR-coord smoothness reuses
:func:`biomevae.models.phylodiva_hyp_philrvae.philr_coord_smoothness`.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.diva_hyp_philrvae import DIVAHyperbolicPhILRVAE
from biomevae.models.philrvae import DataKind, LikelihoodName, TaxonomyGraph
from biomevae.models.phylodiva_hyp_philrvae import (
    build_philr_parent_contrast_index,
    philr_coord_smoothness,
)
from biomevae.models.taxi_treedtmvae import (
    ConditionalStudyCritic,
    _valid_class_mask,
    conditional_coral_by_class_domain,
    conditional_orthogonality_loss,
    make_class_context,
)


__all__ = ["TAXIHyperbolicPhILRVAE"]


class TAXIHyperbolicPhILRVAE(DIVAHyperbolicPhILRVAE):
    """DIVA Hyperbolic PhILR-VAE with taxonomy-protected conditional invariance.

    The base DIVA factors are reinterpreted as ``z_d`` / ``z_tau`` / ``z_rho``;
    the classifier head reads ``[z_tau, z_rho]``; the GRL'd conditional
    study critic only sees ``z_rho``.
    """

    def __init__(
        self,
        *,
        n_domains: int,
        n_classes: int,
        taxg: TaxonomyGraph,
        curvature: float = 1.0,
        hidden: Optional[Sequence[int]] = None,
        latent_d: int = 4,
        latent_tau: int = 8,
        latent_rho: int = 8,
        dropout: float = 0.1,
        aux_hidden: int = 64,
        critic_hidden: int = 128,
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
        grl_lambda: float = 1.0,
        edge_length: Optional[np.ndarray] = None,
        condition_orth_on_domain: bool = True,
    ) -> None:
        super().__init__(
            n_domains=n_domains,
            n_classes=n_classes,
            taxg=taxg,
            curvature=curvature,
            hidden=hidden,
            latent_d=latent_d,
            latent_y=latent_tau,
            latent_x=latent_rho,
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

        self.latent_tau = int(latent_tau)
        self.latent_rho = int(latent_rho)
        self.condition_orth_on_domain = bool(condition_orth_on_domain)

        # Combined [z_tau, z_rho] prediction head.
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

        # PhILR-coord smoothness scaffolding (identical to PhyloDIVA).
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

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    def set_grl_lambda(self, value: float) -> None:
        self.conditional_critic.set_lambda(float(value))

    def z_tau(self, out: Dict[str, torch.Tensor], *, use_mean: bool = True) -> torch.Tensor:
        return out["mu_y"] if use_mean else out["z_y_tan"]

    def z_rho(self, out: Dict[str, torch.Tensor], *, use_mean: bool = True) -> torch.Tensor:
        return out["mu_x"] if use_mean else out["z_x_tan"]

    def z_domain(self, out: Dict[str, torch.Tensor], *, use_mean: bool = True) -> torch.Tensor:
        return out["mu_d"] if use_mean else out["z_d_tan"]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        domain: Optional[torch.Tensor] = None,
        klass: Optional[torch.Tensor] = None,
        *,
        data_kind: DataKind = "relative",
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(x, domain=domain, klass=klass, data_kind=data_kind)

        # Preserve the z_tau-only logits for the adaptation context and the
        # optional tau-aux loss.
        out["tau_class_logits"] = out["class_logits"]

        class_latent = torch.cat(
            [
                self.z_tau(out, use_mean=not self.aux_on_sample),
                self.z_rho(out, use_mean=not self.aux_on_sample),
            ],
            dim=-1,
        )
        out["class_logits"] = self.taxi_class_head(class_latent)
        return out

    @torch.no_grad()
    def predict_class(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
    ) -> torch.Tensor:
        return self.forward(x, data_kind=data_kind)["class_logits"]

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
        lambda_philr_smooth: float = 0.0,
        lambda_orth: float = 0.0,
        lambda_tau_aux: float = 0.0,
        adversary_on_mean: bool = True,
        coral_on_mean: bool = True,
        orth_on_mean: bool = True,
        min_coral_weight: float = 2.0,
        min_orth_weight: float = 4.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = out["mu_y"].device
        domain = domain.to(device).long()

        z_tau_adv = self.z_tau(out, use_mean=adversary_on_mean)
        z_rho_adv = self.z_rho(out, use_mean=adversary_on_mean)

        z_tau_coral = self.z_tau(out, use_mean=coral_on_mean)
        z_rho_coral = self.z_rho(out, use_mean=coral_on_mean)

        z_tau_orth = self.z_tau(out, use_mean=orth_on_mean)
        z_rho_orth = self.z_rho(out, use_mean=orth_on_mean)

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
                z_rho_adv, z_tau_adv, y_context, domain,
            )
            scaled = float(lambda_cond_critic) * critic
            total = total + scaled
            metrics["cond_critic"] = critic.detach()
            metrics["cond_critic_scaled"] = scaled.detach()

        if float(lambda_cond_coral) > 0.0:
            coral = conditional_coral_by_class_domain(
                z_rho_coral, domain, y_context,
                n_domains=self.n_domains,
                min_weight=float(min_coral_weight),
            )
            scaled = float(lambda_cond_coral) * coral
            total = total + scaled
            metrics["cond_coral"] = coral.detach()
            metrics["cond_coral_scaled"] = scaled.detach()

        if float(lambda_philr_smooth) > 0.0:
            smooth = philr_coord_smoothness(
                out["coord_mu"],
                self.parent_contrast_idx,
                self.edge_length,
            )
            scaled = float(lambda_philr_smooth) * smooth
            total = total + scaled
            metrics["philr_smooth"] = smooth.detach()
            metrics["philr_smooth_scaled"] = scaled.detach()

        if float(lambda_orth) > 0.0:
            orth = conditional_orthogonality_loss(
                z_tau_orth, z_rho_orth, domain, y_context,
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
                    out["tau_class_logits"][valid_y], klass[valid_y],
                )
            else:
                tau_aux = torch.zeros((), device=device)
            scaled = float(lambda_tau_aux) * tau_aux
            total = total + scaled
            metrics["tau_aux"] = tau_aux.detach()
            metrics["tau_aux_scaled"] = scaled.detach()

        metrics.setdefault("cond_critic", torch.zeros((), device=device))
        metrics.setdefault("cond_coral", torch.zeros((), device=device))
        metrics.setdefault("philr_smooth", torch.zeros((), device=device))
        metrics.setdefault("orth", torch.zeros((), device=device))
        metrics.setdefault("tau_aux", torch.zeros((), device=device))

        return total, metrics

    # ------------------------------------------------------------------
    # Full loss
    # ------------------------------------------------------------------

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
        lambda_cond_critic: float = 0.0,
        lambda_cond_coral: float = 0.0,
        lambda_philr_smooth: float = 0.0,
        lambda_orth: float = 0.0,
        lambda_tau_aux: float = 0.0,
        adversary_on_mean: bool = True,
        coral_on_mean: bool = True,
        orth_on_mean: bool = True,
        min_coral_weight: float = 2.0,
        min_orth_weight: float = 4.0,
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
            lambda_cond_critic=lambda_cond_critic,
            lambda_cond_coral=lambda_cond_coral,
            lambda_philr_smooth=lambda_philr_smooth,
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
