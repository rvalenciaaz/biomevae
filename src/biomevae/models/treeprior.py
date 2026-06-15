from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .graph import _ensure_hidden_list, prepare_graph_kwargs
from .taxonomy_graph import TaxonomyGraphEncoder
from .vae import get_activation


class TreeStructuredPriorVAE(nn.Module):
    # Clamp range for the learnable ``prior_logvar`` parameters.  A lower
    # bound of -2.0 keeps the prior standard deviation above
    # ``exp(-1.0) ≈ 0.37``, which stops the conditional prior from
    # collapsing onto a delta function (and trivially matching the
    # posterior — the classical failure mode of learnable priors that
    # manifests as posterior collapse downstream).
    _PRIOR_LOGVAR_MIN = -2.0
    _PRIOR_LOGVAR_MAX = 2.0

    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        dropout: float,
        activation: str,
        layer_norm: bool,
        graph_spec: Dict[str, Any],
        gnn_hidden: Sequence[int],
        gnn_dropout: float = 0.0,
        graph_mode: str | None = None,
        gnn_type: str | None = None,
        prior_sigma: float = 1.0,
        branch_reg: float = 0.0,
    ) -> None:
        super().__init__()
        self.graph_encoder = TaxonomyGraphEncoder(
            graph_spec=graph_spec,
            hidden_dims=_ensure_hidden_list(gnn_hidden),
            activation=activation,
            dropout=gnn_dropout,
        )

        rep_dim = self.graph_encoder.output_dim
        enc_layers = []
        prev = rep_dim
        for h in hidden:
            layer = nn.Linear(prev, h)
            enc_layers.append(layer)
            if layer_norm:
                enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(get_activation(activation))
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers = []
        prev_dec = latent_dim
        for h in reversed(list(hidden)):
            dec_layers.append(nn.Linear(prev_dec, h))
            if layer_norm:
                dec_layers.append(nn.LayerNorm(h))
            dec_layers.append(get_activation(activation))
            if dropout > 0.0:
                dec_layers.append(nn.Dropout(dropout))
            prev_dec = h
        dec_layers.append(nn.Linear(prev_dec, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        node_count = int(graph_spec["num_nodes"])
        self.prior_mu = nn.Parameter(torch.zeros(node_count, latent_dim))
        # Initialise ``prior_logvar`` near ``2*log(prior_sigma)`` and clamp
        # it inside the safe region so the starting point is valid.
        init_logvar = torch.log(torch.full((node_count, latent_dim), prior_sigma ** 2))
        init_logvar = init_logvar.clamp(self._PRIOR_LOGVAR_MIN, self._PRIOR_LOGVAR_MAX)
        self.prior_logvar = nn.Parameter(init_logvar)

        edges = torch.tensor(
            [[int(e[0]), int(e[1])] for e in graph_spec.get("edges", [])],
            dtype=torch.long,
        )
        weights = torch.tensor(
            [float(e[2]) for e in graph_spec.get("edges", [])],
            dtype=torch.float32,
        )
        if edges.numel() == 0:
            edges = torch.zeros((0, 2), dtype=torch.long)
            weights = torch.zeros((0,), dtype=torch.float32)
        self.register_buffer("edge_index", edges)
        self.register_buffer("edge_weight", weights)
        self.branch_reg = float(branch_reg)

        # ``_prior_blend`` linearly mixes a fixed N(0, prior_sigma²) prior
        # (blend=0) with the adaptive conditional prior (blend=1).  The
        # trainer ramps this alongside the KL warmup so that the encoder
        # can learn an informative q(z|x) against a stable, non-flexible
        # reference first — this is a principled, training-wheel fix for
        # the posterior collapse that learnable priors routinely induce.
        self.register_buffer(
            "_prior_blend", torch.tensor(1.0, dtype=torch.float32)
        )
        self._prior_sigma_sq = float(prior_sigma) ** 2

    def set_prior_blend(self, blend: float) -> None:
        """Mixing coefficient between N(0, σ²I) and the conditional prior.

        ``blend=0`` → pure N(0, σ²I).  ``blend=1`` → fully adaptive
        conditional prior as originally designed.  Intermediate values
        linearly interpolate the prior mean/variance on each sample.
        """
        value = float(max(0.0, min(1.0, blend)))
        self._prior_blend.fill_(value)

    def encode(self, x: torch.Tensor):
        pooled = self.graph_encoder.sample_representation(x)
        h = self.encoder(pooled)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _clamped_prior_logvar(self) -> torch.Tensor:
        return self.prior_logvar.clamp(
            self._PRIOR_LOGVAR_MIN, self._PRIOR_LOGVAR_MAX
        )

    def conditional_prior(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        selector = self.graph_encoder.feature_selector
        # Tree-structured priors are applied at the feature level by
        # selecting the taxonomy-linked nodes that correspond to the input
        # feature indices.  We clamp ``prior_logvar`` so the prior can't
        # shrink below a floor (prevents trivial posterior matching).
        prior_logvar_safe = self._clamped_prior_logvar()
        feature_mu = self.prior_mu.index_select(0, selector)
        feature_logvar = prior_logvar_safe.index_select(0, selector)
        # Detach the per-sample weighting from the encoder's gradient:
        # otherwise the prior-matching pressure back-propagates into the
        # graph encoder and can actively drive posterior collapse.
        weights = self.graph_encoder.normalize_weights(x).detach()
        cond_mu = weights @ feature_mu
        cond_var = weights @ torch.exp(feature_logvar)
        # Floor the per-sample prior variance at 0.25 (std ≥ 0.5): a much
        # stronger safeguard than the original ``1e-6`` clamp, which still
        # allowed near-degenerate priors.
        cond_var = torch.clamp(cond_var, min=0.25)

        # Blend with a fixed N(0, σ²I) reference.  During the fragile
        # early-training phase the caller sets ``_prior_blend=0`` (pure
        # reference prior); it's then ramped to 1 over the KL warmup.
        blend = float(self._prior_blend.item())
        if blend < 1.0:
            ref_var = torch.full_like(cond_var, self._prior_sigma_sq)
            ref_mu = torch.zeros_like(cond_mu)
            prior_mu = blend * cond_mu + (1.0 - blend) * ref_mu
            prior_var = blend * cond_var + (1.0 - blend) * ref_var
        else:
            prior_mu = cond_mu
            prior_var = cond_var

        prior_logvar = torch.log(prior_var.clamp(min=1e-4))

        reg = torch.tensor(0.0, device=prior_mu.device)
        if self.branch_reg > 0.0 and self.edge_index.numel() > 0:
            diffs_mu = self.prior_mu.index_select(
                0, self.edge_index[:, 0]
            ) - self.prior_mu.index_select(0, self.edge_index[:, 1])
            diffs_log = prior_logvar_safe.index_select(
                0, self.edge_index[:, 0]
            ) - prior_logvar_safe.index_select(0, self.edge_index[:, 1])
            edge_pen = (diffs_mu.pow(2).sum(dim=1) + diffs_log.pow(2).sum(dim=1)) * self.edge_weight
            reg = self.branch_reg * edge_pen.mean()

        return {"mu": prior_mu, "logvar": prior_logvar, "regularizer": reg}

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def prepare_tree_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = prepare_graph_kwargs(kwargs)
    out.pop("prior_kind", None)
    out["branch_reg"] = float(out.get("branch_reg", 0.0))
    out["prior_sigma"] = float(out.get("prior_sigma", 1.0))
    return out
