"""DIVA wrapper for the Hyperbolic PhILR-VAE backbone.

Mirrors :class:`biomevae.models.diva_treedtmvae.DIVATreeDTMVAE` exactly:

* Three independent ILR -> tangent encoders for ``z_d``, ``z_y``, ``z_x``,
  each living on its own Poincaré sub-manifold.
* Conditional Gaussian priors :math:`p(z_d|d)`, :math:`p(z_y|y)`, and
  :math:`N(0, I)` on :math:`z_x` (in tangent space).
* Semi-supervised marginalisation over unlabelled ``y`` via
  :math:`q(y|x)` from the class classifier; categorical KL against the
  empirical class prior.
* Reconstruction with one of the compositional likelihoods exposed by
  :class:`biomevae.models.philrvae.PhILRVAE`
  (``philr_gaussian`` / ``multinomial`` / ``dirichlet_multinomial`` /
  ``dirichlet_tree_multinomial`` / ``dirichlet_tree``).

Inputs are samples-by-leaves tensors in ``taxg.leaf_ids`` order. The
backbone class :class:`HyperbolicPhILRVAE` already projects ball points
back to tangent space via ``logmap0`` before any Euclidean layer
(audit D2).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.diva_treedtmvae import (
    ConditionalGaussianPrior,
    DIVAComponents,
    MLPClassifier,
    diag_gaussian_kl,
)
from biomevae.models.hyperbolic_philrvae import HyperbolicPhILRVAE
from biomevae.models.philrvae import (
    DataKind,
    LikelihoodName,
    TaxonomyGraph,
    _MLP,
)


__all__ = ["DIVAHyperbolicPhILRVAE"]


class _PhILRBalanceEncoder(nn.Module):
    """ILR-trunk encoder producing tangent-space (mu, logvar)."""

    def __init__(
        self,
        philr_module: nn.Module,
        latent_dim: int,
        *,
        hidden: Sequence[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.philr = philr_module
        self.trunk = _MLP(
            self.philr.n_coords,
            hidden,
            out_dim=None,
            dropout=dropout,
            layer_norm=True,
        )
        self.fc_mu = nn.Linear(self.trunk.out_dim, int(latent_dim))
        self.fc_logvar = nn.Linear(self.trunk.out_dim, int(latent_dim))
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = self.philr(x, data_kind=data_kind)
        h = self.trunk(coords)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar


class DIVAHyperbolicPhILRVAE(HyperbolicPhILRVAE):
    """DIVA-factorised PhILR-VAE with Poincaré-ball latent."""

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
    ) -> None:
        if min(latent_d, latent_y, latent_x) <= 0:
            raise ValueError("latent_d, latent_y and latent_x must all be positive.")

        self.latent_d = int(latent_d)
        self.latent_y = int(latent_y)
        self.latent_x = int(latent_x)
        self.total_latent = int(latent_d + latent_y + latent_x)
        self.n_domains = int(n_domains)
        self.n_classes = int(n_classes)
        self.aux_on_sample = bool(aux_on_sample)

        if hidden is None:
            hidden = (256, 128)

        super().__init__(
            taxg,
            latent_dim=self.total_latent,
            curvature=curvature,
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

        for attr in ("encoder_trunk", "fc_mu", "fc_logvar"):
            if hasattr(self, attr):
                delattr(self, attr)

        self.enc_d = _PhILRBalanceEncoder(
            self.philr, latent_d, hidden=hidden, dropout=dropout,
        )
        self.enc_y = _PhILRBalanceEncoder(
            self.philr, latent_y, hidden=hidden, dropout=dropout,
        )
        self.enc_x = _PhILRBalanceEncoder(
            self.philr, latent_x, hidden=hidden, dropout=dropout,
        )

        self.prior_d = ConditionalGaussianPrior(n_domains, self.latent_d)
        self.prior_y = ConditionalGaussianPrior(n_classes, self.latent_y)

        self.domain_classifier = MLPClassifier(
            self.latent_d, n_domains, hidden=aux_hidden, dropout=dropout,
        )
        self.class_classifier = MLPClassifier(
            self.latent_y, n_classes, hidden=aux_hidden, dropout=dropout,
        )

        if class_prior is None:
            prior = torch.full((n_classes,), 1.0 / float(n_classes))
        else:
            prior = class_prior.detach().float()
            if prior.numel() != n_classes:
                raise ValueError("class_prior must have length n_classes.")
            if (prior < 0).any() or prior.sum() <= 0:
                raise ValueError("class_prior must be non-negative and have positive sum.")
            prior = prior / prior.sum()

        self.register_buffer("class_prior_log_probs", prior.clamp_min(1e-12).log())

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
    ) -> Dict[str, torch.Tensor]:
        mu_d, lv_d = self.enc_d(x, data_kind=data_kind)
        mu_y, lv_y = self.enc_y(x, data_kind=data_kind)
        mu_x, lv_x = self.enc_x(x, data_kind=data_kind)

        return {
            "mu_d": mu_d, "lv_d": lv_d,
            "mu_y": mu_y, "lv_y": lv_y,
            "mu_x": mu_x, "lv_x": lv_x,
        }

    def sample_latents(self, enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Tangent samples per factor; concatenate and map to a single ball point.
        v_d = enc["mu_d"] + torch.randn_like(enc["mu_d"]) * torch.exp(0.5 * enc["lv_d"].clamp(-30.0, 20.0))
        v_y = enc["mu_y"] + torch.randn_like(enc["mu_y"]) * torch.exp(0.5 * enc["lv_y"].clamp(-30.0, 20.0))
        v_x = enc["mu_x"] + torch.randn_like(enc["mu_x"]) * torch.exp(0.5 * enc["lv_x"].clamp(-30.0, 20.0))
        v = torch.cat([v_d, v_y, v_x], dim=-1)
        z = self.manifold.projx(self.manifold.expmap0(v))
        return {
            "z_d_tan": v_d, "z_y_tan": v_y, "z_x_tan": v_x,
            "v": v, "z": z,
        }

    def latent_split(self, z_tan: torch.Tensor) -> Dict[str, torch.Tensor]:
        d, y = self.latent_d, self.latent_y
        return {
            "z_d": z_tan[..., :d],
            "z_y": z_tan[..., d : d + y],
            "z_x": z_tan[..., d + y :],
        }

    def decode_parts(
        self,
        mu_d: torch.Tensor,
        mu_y: torch.Tensor,
        mu_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        v = torch.cat([mu_d, mu_y, mu_x], dim=-1)
        z = self.manifold.projx(self.manifold.expmap0(v))
        return self.decode(z)

    def forward(
        self,
        x: torch.Tensor,
        domain: Optional[torch.Tensor] = None,
        klass: Optional[torch.Tensor] = None,
        *,
        data_kind: DataKind = "relative",
    ) -> Dict[str, torch.Tensor]:
        obs_coords = self.philr(x, data_kind=data_kind)
        enc = self.encode(x, data_kind=data_kind)
        lat = self.sample_latents(enc)
        dec = self.decode(lat["z"])

        aux_d_in = lat["z_d_tan"] if self.aux_on_sample else enc["mu_d"]
        aux_y_in = lat["z_y_tan"] if self.aux_on_sample else enc["mu_y"]

        out: Dict[str, torch.Tensor] = {
            "obs_coords": obs_coords,
            **enc,
            **lat,
            **dec,
            "domain_logits": self.domain_classifier(aux_d_in),
            "class_logits": self.class_classifier(aux_y_in),
            "y_nodes": x,
        }
        out["mu_z"] = lat["v"]
        out["logvar_z"] = torch.cat(
            [enc["lv_d"], enc["lv_y"], enc["lv_x"]], dim=-1,
        )

        if domain is not None:
            out["domain"] = domain
        if klass is not None:
            out["klass"] = klass

        return out

    @torch.no_grad()
    def predict_class(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
    ) -> torch.Tensor:
        enc = self.encode(x, data_kind=data_kind)
        return self.class_classifier(enc["mu_y"])

    @torch.no_grad()
    def reconstruct_leaf_proportions(
        self,
        x: torch.Tensor,
        *,
        data_kind: DataKind = "relative",
        use_mean: bool = True,
        zero_domain: bool = False,
        zero_residual: bool = False,
    ) -> torch.Tensor:
        enc = self.encode(x, data_kind=data_kind)

        if use_mean:
            z_d, z_y, z_x = enc["mu_d"], enc["mu_y"], enc["mu_x"]
        else:
            sampled = self.sample_latents(enc)
            z_d = sampled["z_d_tan"]
            z_y = sampled["z_y_tan"]
            z_x = sampled["z_x_tan"]

        if zero_domain:
            z_d = torch.zeros_like(z_d)
        if zero_residual:
            z_x = torch.zeros_like(z_x)

        return self.decode_parts(z_d, z_y, z_x)["leaf_prob"]

    # ------------------------------------------------------------------
    # DIVA objective terms (mirror DIVATreeDTMVAE)
    # ------------------------------------------------------------------

    def _domain_mask(self, domain: torch.Tensor) -> torch.Tensor:
        return (domain >= 0) & (domain < self.n_domains)

    def _class_mask(self, klass: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if klass is None:
            return None
        return (klass >= 0) & (klass < self.n_classes)

    def compute_diva_components(
        self,
        out: Dict[str, torch.Tensor],
        domain: torch.Tensor,
        klass: Optional[torch.Tensor],
        *,
        free_bits: float = 0.0,
        use_unlabelled_class_marginal: bool = True,
    ) -> DIVAComponents:
        device = out["mu_x"].device
        batch = out["mu_x"].size(0)

        domain = domain.to(device).long()
        domain_mask = self._domain_mask(domain)
        if not bool(domain_mask.all()):
            raise ValueError("DIVA requires a valid domain label for every training sample.")

        kl_d = torch.zeros(batch, device=device)
        kl_d[domain_mask] = self.prior_d.kl_to_label(
            out["mu_d"][domain_mask],
            out["lv_d"][domain_mask],
            domain[domain_mask],
            free_bits=free_bits,
        )
        ce_d = F.cross_entropy(out["domain_logits"][domain_mask], domain[domain_mask])

        kl_x = diag_gaussian_kl(out["mu_x"], out["lv_x"], free_bits=free_bits)

        kl_y = torch.zeros(batch, device=device)
        categorical_y_kl = torch.zeros(batch, device=device)
        ce_y = torch.zeros((), device=device)
        n_y_labelled = 0

        class_logits = out["class_logits"]

        if klass is not None:
            klass = klass.to(device).long()
            y_mask = self._class_mask(klass)
        else:
            y_mask = torch.zeros(batch, dtype=torch.bool, device=device)

        if y_mask.any():
            n_y_labelled = int(y_mask.sum().item())
            kl_y[y_mask] = self.prior_y.kl_to_label(
                out["mu_y"][y_mask],
                out["lv_y"][y_mask],
                klass[y_mask],
                free_bits=free_bits,
            )
            ce_y = F.cross_entropy(class_logits[y_mask], klass[y_mask])

        unlabelled = ~y_mask

        if use_unlabelled_class_marginal and unlabelled.any():
            probs = F.softmax(class_logits[unlabelled], dim=1)
            log_probs = F.log_softmax(class_logits[unlabelled], dim=1)

            kl_all = self.prior_y.kl_to_all(
                out["mu_y"][unlabelled],
                out["lv_y"][unlabelled],
                free_bits=free_bits,
            )
            kl_y[unlabelled] = (probs * kl_all).sum(dim=1)

            prior_logp = self.class_prior_log_probs.to(device).unsqueeze(0)
            categorical_y_kl[unlabelled] = (probs * (log_probs - prior_logp)).sum(dim=1)

        elif unlabelled.any():
            kl_y[unlabelled] = diag_gaussian_kl(
                out["mu_y"][unlabelled],
                out["lv_y"][unlabelled],
                free_bits=free_bits,
            )

        return DIVAComponents(
            kl_d=kl_d,
            kl_y=kl_y,
            kl_x=kl_x,
            categorical_y_kl=categorical_y_kl,
            ce_d=ce_d,
            ce_y=ce_y,
            n_domain_labelled=int(domain_mask.sum().item()),
            n_y_labelled=n_y_labelled,
        )

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = (
            self.forward(x, domain=domain, klass=klass, data_kind=data_kind)
            if out is None else out
        )
        likelihood = self.default_likelihood if likelihood is None else likelihood

        recon = self.reconstruction_nll(
            x, out,
            likelihood=likelihood,
            data_kind=data_kind,
            validate_counts=validate_counts,
            observation_pseudocount=observation_pseudocount,
        )

        diva = self.compute_diva_components(
            out, domain, klass,
            free_bits=free_bits,
            use_unlabelled_class_marginal=use_unlabelled_class_marginal,
        )

        kl_total = diva.kl_d + diva.kl_y + diva.kl_x

        uses_tree_concentration = likelihood in {
            "dirichlet_tree", "dirichlet_tree_multinomial",
        }
        reg = (
            self.concentration_regularization() * float(concentration_l2)
            if uses_tree_concentration else x.new_zeros(())
        )

        total = (
            recon.mean()
            + float(beta) * kl_total.mean()
            + float(alpha_d) * diva.ce_d
            + float(alpha_y) * diva.ce_y
            + float(unlabelled_y_prior_weight) * diva.categorical_y_kl.mean()
            + reg
        )

        metrics = {
            "loss": total.detach(),
            "reconstruction_nll": recon.mean().detach(),
            "kl_total": kl_total.mean().detach(),
            "kl_d": diva.kl_d.mean().detach(),
            "kl_y": diva.kl_y.mean().detach(),
            "kl_x": diva.kl_x.mean().detach(),
            "ce_d": diva.ce_d.detach(),
            "ce_y": diva.ce_y.detach(),
            "categorical_y_kl": diva.categorical_y_kl.mean().detach(),
            "n_y_labelled": torch.tensor(float(diva.n_y_labelled), device=x.device),
            "coord_scale_mean": self.coord_scale().mean().detach(),
            "concentration_l2": reg.detach(),
        }
        if self.group_depth.numel() > 0:
            metrics["group_concentration_mean"] = self.group_concentration().mean().detach()

        return total, metrics
