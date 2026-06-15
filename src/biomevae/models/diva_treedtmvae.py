"""DIVA wrapper for a tree-structured Dirichlet-tree VAE.

This module is the statistically appropriate replacement for the old
``diva_treenbvae.py`` wrapper.  It assumes that the backbone module
``tree_dtm_vae.py`` provides the tree decoder and likelihoods:

* ``dirichlet_tree_multinomial`` for integer count tables.
* ``tree_multinomial`` for integer count tables without overdispersion.
* ``dirichlet_tree`` for closed relative-abundance compositions.

The model input is ``node_values`` with shape ``(batch, n_tree_nodes)``:
leaf values plus internal sums/proportions, as returned by
``build_treevae_dataset`` in ``tree_dtm_vae.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from biomevae.models.tree_dtm_vae import (
    LikelihoodName,
    TreeBalanceEncoder,
    TreeDTMVAE,
    TreeTopology,
)


__all__ = [
    "ConditionalGaussianPrior",
    "DIVAComponents",
    "DIVATreeDTMVAE",
    "diag_gaussian_kl",
]


def diag_gaussian_kl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_mu: Optional[torch.Tensor] = None,
    prior_logvar: Optional[torch.Tensor] = None,
    *,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """KL[N(mu, var) || N(prior_mu, prior_var)] per sample.

    ``prior_mu`` and ``prior_logvar`` may be ``None`` for the standard normal
    or tensors broadcastable to ``mu``.
    """
    logvar = logvar.clamp(min=-30.0, max=20.0)

    if prior_mu is None:
        prior_mu = torch.zeros_like(mu)
    if prior_logvar is None:
        prior_logvar = torch.zeros_like(logvar)

    prior_logvar = prior_logvar.clamp(min=-30.0, max=20.0)
    var = logvar.exp()
    prior_var = prior_logvar.exp()

    per_dim = 0.5 * (
        prior_logvar
        - logvar
        + (var + (mu - prior_mu).pow(2)) / prior_var.clamp_min(1e-12)
        - 1.0
    )

    if free_bits > 0.0:
        per_dim = per_dim.clamp_min(float(free_bits))

    return per_dim.sum(dim=-1)


class ConditionalGaussianPrior(nn.Module):
    """Label-conditioned diagonal Gaussian prior.

    The embedding stores a mean and log-variance for each domain/class.
    It implements the DIVA conditional priors ``p(z_d | d)`` and
    ``p(z_y | y)``.
    """

    def __init__(self, n_conditions: int, latent_dim: int, *, init_logvar: float = 0.0) -> None:
        super().__init__()
        if n_conditions <= 0:
            raise ValueError("n_conditions must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.n_conditions = int(n_conditions)
        self.latent_dim = int(latent_dim)

        self.mean = nn.Embedding(self.n_conditions, self.latent_dim)
        self.logvar = nn.Embedding(self.n_conditions, self.latent_dim)

        nn.init.zeros_(self.mean.weight)
        nn.init.constant_(self.logvar.weight, float(init_logvar))

    def forward(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = labels.long()
        if labels.numel() and ((labels < 0).any() or (labels >= self.n_conditions).any()):
            raise ValueError("Condition labels are outside the prior's range.")

        return self.mean(labels), self.logvar(labels).clamp(-10.0, 10.0)

    def kl_to_label(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        labels: torch.Tensor,
        *,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        prior_mu, prior_logvar = self.forward(labels)
        return diag_gaussian_kl(mu, logvar, prior_mu, prior_logvar, free_bits=free_bits)

    def kl_to_all(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        """KL to every condition-specific prior.

        Returns a tensor with shape ``(batch, n_conditions)``.
        """
        b = mu.size(0)
        c = self.n_conditions

        mu_e = mu[:, None, :].expand(b, c, self.latent_dim)
        lv_e = logvar[:, None, :].expand(b, c, self.latent_dim)

        prior_mu = self.mean.weight[None, :, :].expand_as(mu_e)
        prior_lv = self.logvar.weight[None, :, :].expand_as(lv_e).clamp(-10.0, 10.0)

        return diag_gaussian_kl(mu_e, lv_e, prior_mu, prior_lv, free_bits=free_bits)


class MLPClassifier(nn.Module):
    """Small latent-space classifier used for DIVA auxiliary objectives."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        *,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@dataclass
class DIVAComponents:
    """Container for per-sample DIVA terms."""

    kl_d: torch.Tensor
    kl_y: torch.Tensor
    kl_x: torch.Tensor
    categorical_y_kl: torch.Tensor
    ce_d: torch.Tensor
    ce_y: torch.Tensor
    n_domain_labelled: int
    n_y_labelled: int


class DIVATreeDTMVAE(TreeDTMVAE):
    """DIVA-factorized VAE with a tree-structured microbiome likelihood.

    The model uses three separate tree-balance encoders:

    * ``z_d``: domain/study latent, regularized by ``p(z_d | domain)``.
    * ``z_y``: class/phenotype latent, regularized by ``p(z_y | class)``.
    * ``z_x``: residual latent, regularized by ``N(0, I)``.

    The concatenated latent ``[z_d, z_y, z_x]`` is decoded by the
    tree-softmax decoder from ``TreeDTMVAE``. Reconstruction is scored with
    a tree likelihood, not an independent NB likelihood.
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
        encoder_pseudocount: float = 0.5,
        init_concentration: float = 50.0,
        likelihood: LikelihoodName = "dirichlet_tree_multinomial",
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

        # Initialize the parent class to reuse its decoder, tree buffers and
        # likelihood methods. The single encoder created by the parent is
        # discarded and replaced by three independent DIVA encoders.
        super().__init__(
            topo,
            hidden=hidden,
            latent_dim=self.total_latent,
            encoder_layers=1,
            decoder_hidden=decoder_hidden,
            decoder_layers=decoder_layers,
            dropout=dropout,
            encoder_pseudocount=encoder_pseudocount,
            init_concentration=init_concentration,
            likelihood=likelihood,
        )

        if hasattr(self, "encoder"):
            del self.encoder

        self.enc_d = TreeBalanceEncoder(
            topo,
            hidden=hidden,
            latent_dim=self.latent_d,
            n_layers=encoder_layers,
            dropout=dropout,
            pseudocount=encoder_pseudocount,
        )
        self.enc_y = TreeBalanceEncoder(
            topo,
            hidden=hidden,
            latent_dim=self.latent_y,
            n_layers=encoder_layers,
            dropout=dropout,
            pseudocount=encoder_pseudocount,
        )
        self.enc_x = TreeBalanceEncoder(
            topo,
            hidden=hidden,
            latent_dim=self.latent_x,
            n_layers=encoder_layers,
            dropout=dropout,
            pseudocount=encoder_pseudocount,
        )

        self.prior_d = ConditionalGaussianPrior(n_domains, self.latent_d)
        self.prior_y = ConditionalGaussianPrior(n_classes, self.latent_y)

        self.domain_classifier = MLPClassifier(
            self.latent_d,
            n_domains,
            hidden=aux_hidden,
            dropout=dropout,
        )
        self.class_classifier = MLPClassifier(
            self.latent_y,
            n_classes,
            hidden=aux_hidden,
            dropout=dropout,
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

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar.clamp(-30.0, 20.0))

    def encode(self, node_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu_d, lv_d = self.enc_d(node_values)
        mu_y, lv_y = self.enc_y(node_values)
        mu_x, lv_x = self.enc_x(node_values)

        return {
            "mu_d": mu_d,
            "lv_d": lv_d,
            "mu_y": mu_y,
            "lv_y": lv_y,
            "mu_x": mu_x,
            "lv_x": lv_x,
        }

    def sample_latents(self, enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_d = self.reparam(enc["mu_d"], enc["lv_d"])
        z_y = self.reparam(enc["mu_y"], enc["lv_y"])
        z_x = self.reparam(enc["mu_x"], enc["lv_x"])

        return {
            "z_d": z_d,
            "z_y": z_y,
            "z_x": z_x,
            "z": torch.cat([z_d, z_y, z_x], dim=-1),
        }

    def latent_split(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        d, y = self.latent_d, self.latent_y
        return {
            "z_d": z[..., :d],
            "z_y": z[..., d : d + y],
            "z_x": z[..., d + y :],
        }

    def decode_parts(
        self,
        z_d: torch.Tensor,
        z_y: torch.Tensor,
        z_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.decode(torch.cat([z_d, z_y, z_x], dim=-1))

    def forward(
        self,
        node_values: torch.Tensor,
        domain: Optional[torch.Tensor] = None,
        klass: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode(node_values)
        lat = self.sample_latents(enc)
        dec = self.decode(lat["z"])

        aux_d_in = lat["z_d"] if self.aux_on_sample else enc["mu_d"]
        aux_y_in = lat["z_y"] if self.aux_on_sample else enc["mu_y"]

        out: Dict[str, torch.Tensor] = {
            **enc,
            **lat,
            **dec,
            "domain_logits": self.domain_classifier(aux_d_in),
            "class_logits": self.class_classifier(aux_y_in),
            "y_nodes": node_values,
        }

        if domain is not None:
            out["domain"] = domain
        if klass is not None:
            out["klass"] = klass

        return out

    @torch.no_grad()
    def predict_class(self, node_values: torch.Tensor) -> torch.Tensor:
        enc = self.encode(node_values)
        return self.class_classifier(enc["mu_y"])

    @torch.no_grad()
    def reconstruct_leaf_proportions(
        self,
        node_values: torch.Tensor,
        *,
        use_mean: bool = True,
        zero_domain: bool = False,
        zero_residual: bool = False,
    ) -> torch.Tensor:
        enc = self.encode(node_values)

        if use_mean:
            z_d, z_y, z_x = enc["mu_d"], enc["mu_y"], enc["mu_x"]
        else:
            sampled = self.sample_latents(enc)
            z_d, z_y, z_x = sampled["z_d"], sampled["z_y"], sampled["z_x"]

        if zero_domain:
            z_d = torch.zeros_like(z_d)
        if zero_residual:
            z_x = torch.zeros_like(z_x)

        return self.decode_parts(z_d, z_y, z_x)["leaf_prob"]

    # ------------------------------------------------------------------
    # DIVA objective terms
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
            # Fall back to a standard normal prior if the semi-supervised
            # class marginalization is disabled.
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

    def reconstruction_nll(
        self,
        node_values: torch.Tensor,
        out: Dict[str, torch.Tensor],
        *,
        likelihood: Optional[LikelihoodName] = None,
        validate_counts: bool = True,
        observation_pseudocount: float = 1e-6,
    ) -> torch.Tensor:
        likelihood = self.default_likelihood if likelihood is None else likelihood

        if likelihood == "tree_multinomial":
            return self.tree_multinomial_nll(
                node_values,
                out["edge_log_prob"],
                validate_counts=validate_counts,
            )

        if likelihood == "dirichlet_tree_multinomial":
            return self.dirichlet_tree_multinomial_nll(
                node_values,
                out["edge_log_prob"],
                out["group_concentration"],
                validate_counts=validate_counts,
            )

        if likelihood == "dirichlet_tree":
            return self.dirichlet_tree_nll(
                node_values,
                out["edge_log_prob"],
                out["group_concentration"],
                observation_pseudocount=observation_pseudocount,
            )

        raise ValueError(f"Unknown likelihood {likelihood!r}.")

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(node_values, domain=domain, klass=klass) if out is None else out

        recon = self.reconstruction_nll(
            node_values,
            out,
            likelihood=likelihood,
            validate_counts=validate_counts,
            observation_pseudocount=observation_pseudocount,
        )

        diva = self.compute_diva_components(
            out,
            domain,
            klass,
            free_bits=free_bits,
            use_unlabelled_class_marginal=use_unlabelled_class_marginal,
        )

        kl_total = diva.kl_d + diva.kl_y + diva.kl_x
        reg = self.decoder.concentration_regularization() * float(concentration_l2)

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
            "n_y_labelled": torch.tensor(float(diva.n_y_labelled), device=node_values.device),
            "concentration_l2": reg.detach(),
            "mean_concentration": out["group_concentration"].mean().detach(),
        }

        return total, metrics
