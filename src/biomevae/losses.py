import math

import torch
import torch.nn.functional as F

__all__ = [
    "reconstruction_loss",
    "kl_per_sample",
    "beta_schedule",
    "capacity_schedule",
    "compute_losses",
    "nb_nll",
    "zinb_nll",
    "dirichlet_nll",
    "cyclical_beta_schedule",
    "gaussian_kl",
    "focal_ce_balanced",
    "supcon_loss",
    "effective_number_class_weights",
]


def nb_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_theta: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative Binomial NLL with mean-dispersion parameterisation.

    Uses the lgamma formulation for numerical stability.  The loss is summed
    over features and averaged over the batch so that its scale is comparable
    to the KL term (summed over latent dims, averaged over batch).

    Args:
        x: observed counts ``(batch, p)``
        mu: predicted mean ``(batch, p)``, must be > 0
        log_theta: log inverse-dispersion, per feature ``(p,)`` or broadcastable
        eps: small constant for numerical safety
    """
    theta = log_theta.exp().clamp(min=eps)
    mu = mu.clamp(min=eps)
    ll = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + x * (torch.log(mu) - torch.log(theta + mu))
    )
    return -ll.sum(dim=-1).mean()


def zinb_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_theta: torch.Tensor,
    logit_pi: torch.Tensor,
    *,
    eps: float = 1e-8,
    zero_tol: float = 0.0,
) -> torch.Tensor:
    """Zero-Inflated Negative Binomial NLL (scVI-style parameterisation).

    Mixture density:
        P(x = 0) = pi + (1 - pi) * NB(0 | mu, theta)
        P(x > 0) = (1 - pi) * NB(x | mu, theta)

    where ``pi = sigmoid(logit_pi)`` is a per-sample, per-feature
    zero-inflation probability.  The loss is summed over features and
    averaged over the batch to match the scale of ``nb_nll``.

    Args:
        x: observed counts ``(batch, p)``.
        mu: predicted NB mean ``(batch, p)``, must be > 0.
        log_theta: log NB inverse-dispersion, broadcastable to ``(batch, p)``.
        logit_pi: zero-inflation logits ``(batch, p)``.
        eps: numerical floor for log/clamp safety.
        zero_tol: values ``x <= zero_tol`` are treated as observed zeros
            (defaults to strict zero).
    """
    theta = log_theta.exp().clamp(min=eps)
    mu = mu.clamp(min=eps)

    # log NB(x | mu, theta): reused from nb_nll.
    log_theta_mu = torch.log(theta + mu)
    nb_log_prob = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + x * (torch.log(mu) - log_theta_mu)
    )
    # log NB(0 | mu, theta) = theta * (log theta - log(theta + mu)).
    nb_log_prob_zero = theta * (torch.log(theta) - log_theta_mu)

    # log-sigmoid parameterisation for pi / (1-pi) — numerically stable.
    log_pi = F.logsigmoid(logit_pi)       # log(pi)
    log_1mpi = F.logsigmoid(-logit_pi)    # log(1 - pi)

    # Mixture: log P(x=0) = logsumexp(log pi, log(1-pi) + log NB(0)).
    log_p_zero = torch.logsumexp(
        torch.stack([log_pi, log_1mpi + nb_log_prob_zero], dim=-1), dim=-1,
    )
    log_p_nonzero = log_1mpi + nb_log_prob

    is_zero = x <= zero_tol
    log_p = torch.where(is_zero, log_p_zero, log_p_nonzero)
    return -log_p.sum(dim=-1).mean()


def dirichlet_nll(
    x: torch.Tensor,
    mu_x: torch.Tensor,
    log_concentration: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dirichlet NLL for continuous compositional data.

    Reconstruction loss for models whose decoder outputs a vector of
    predicted proportions ``mu_x`` on the simplex (``sum == 1``) and whose
    observations ``x`` are also proportions (or can be mapped to
    proportions). The Dirichlet is the natural likelihood family for the
    simplex; unlike NB-on-floats (see ``nb_nll``) it *is* a proper
    continuous density, so the concentration parameter has a meaningful
    interpretation (larger → tighter around ``mu_x``, smaller → more
    dispersed).

    Parameterisation: ``alpha = concentration * mu_x`` with
    ``concentration = exp(log_concentration)`` a learnable scalar (or
    broadcast-compatible tensor). Because ``sum(mu_x) = 1`` we have
    ``sum(alpha) = concentration``, which simplifies ``lgamma(sum(alpha))``
    to a single per-sample term.

    Args:
        x: observed proportions ``(batch, p)``, expected to lie on or near
            the simplex. Zeros are clamped to ``eps`` and the vector is
            re-normalised inside the function — callers typically pass
            raw counts divided by library size plus a small pseudocount.
        mu_x: predicted proportions ``(batch, p)``, assumed to already
            sum to 1 (it is also re-normalised defensively).
        log_concentration: log of the total concentration, scalar tensor
            or shape broadcast-compatible with ``(batch,)``.
        eps: floor for ``x`` / ``mu_x`` before taking logs.

    Returns:
        scalar: mean over the batch of the per-sample Dirichlet NLL.
    """
    x = x.clamp(min=eps)
    mu_x = mu_x.clamp(min=eps)
    x = x / x.sum(dim=-1, keepdim=True)
    mu_x = mu_x / mu_x.sum(dim=-1, keepdim=True)

    concentration = log_concentration.exp().clamp(min=eps)
    alpha = concentration * mu_x  # (batch, p) via broadcasting

    # log p(x|α) = lgamma(Σα) − Σ lgamma(α_i) + Σ (α_i − 1) log x_i
    # With Σ mu = 1, Σ α = concentration.
    log_norm = torch.lgamma(alpha.sum(dim=-1)) - torch.lgamma(alpha).sum(dim=-1)
    data_term = ((alpha - 1.0) * torch.log(x)).sum(dim=-1)
    ll = log_norm + data_term
    return -ll.mean()


def reconstruction_loss(x, recon, kind="mae", huber_delta=1.0, per_feature="sum"):
    """Per-sample reconstruction loss, averaged over the batch.

    ``per_feature`` controls how the per-element loss is reduced across
    features (the last-but-one axis after the ``view`` flatten):

    * ``"sum"`` (default) matches the canonical VAE ELBO — the negative
      log-likelihood is summed over features, which is the convention
      used by :func:`nb_nll` / :func:`dirichlet_nll` and the KL term in
      :func:`kl_per_sample` (sum over latent dims). Previously this
      function averaged over features, which gave a per-feature-mean
      reconstruction term (~1e-2 on standardised log1p inputs) while
      the KL stayed on a per-latent-dim-sum scale (~latent_dim). The
      resulting O(input_dim/latent_dim) mismatch drove β-VAE / Hyperbolic
      VAE / Tax-aware / Hyp+Tax / TreePrior / PhyloFusion / Graph VAE /
      Vanilla VAE into posterior collapse on MetaCardis_2020_a (PC1
      variance → 100%, val ELBO diverging, active_units → 0). Summing
      matches the established NB/Dirichlet recon scale and lets the
      existing ``--beta-max 0.05`` default produce a healthy balance.
    * ``"mean"`` preserves the legacy per-feature-mean scale and is
      intended for reporting (see ``biomevae.cli.vae_test``) where the
      reader expects a per-feature error metric.
    """
    if kind == "mse":
        per_element = F.mse_loss(recon, x, reduction="none").view(x.size(0), -1)
    elif kind == "mae":
        per_element = F.l1_loss(recon, x, reduction="none").view(x.size(0), -1)
    elif kind == "huber":
        per_element = F.smooth_l1_loss(
            recon, x, beta=huber_delta, reduction="none"
        ).view(x.size(0), -1)
    else:
        raise ValueError(f"Unknown recon kind: {kind}")
    if per_feature == "sum":
        per = per_element.sum(dim=1)
    elif per_feature == "mean":
        per = per_element.mean(dim=1)
    else:
        raise ValueError(
            f"per_feature must be 'sum' or 'mean', got {per_feature!r}"
        )
    return per.mean()

def kl_per_sample(mu, logvar, free_bits=0.0, prior_mu=None, prior_logvar=None):
    mu = mu.float()
    logvar = logvar.float()
    if prior_mu is None:
        prior_mu = torch.zeros_like(mu)
    else:
        prior_mu = prior_mu.float()
    if prior_logvar is None:
        prior_logvar = torch.zeros_like(logvar)
    else:
        prior_logvar = prior_logvar.float()
    logvar = torch.clamp(logvar, min=-30.0, max=20.0)
    prior_logvar = torch.clamp(prior_logvar, min=-30.0, max=20.0)
    var = torch.exp(logvar)
    prior_var = torch.exp(prior_logvar)
    diff = mu - prior_mu
    per_dim = 0.5 * (
        (var + diff.pow(2)) / prior_var - 1.0 + prior_logvar - logvar
    )
    if free_bits > 0:
        per_dim = torch.clamp(per_dim, min=free_bits)
    return per_dim.sum(dim=1)

def beta_schedule(epoch: int, warmup: int, beta_max: float) -> float:
    if warmup <= 0:
        return beta_max
    return float(min(beta_max, beta_max * epoch / max(1, warmup)))

def capacity_schedule(epoch: int, cap_start: float, cap_end: float, cap_epochs: int) -> float:
    if cap_epochs <= 0:
        return cap_end
    t = min(1.0, epoch / float(cap_epochs))
    return cap_start + t * (cap_end - cap_start)

def compute_losses(
    x, recon, mu, logvar,
    recon_kind="mae", huber_delta=1.0,
    objective="beta",
    beta=1.0,
    free_bits=0.0,
    capacity_C=0.0,
    capacity_gamma=1.0,
    prior_mu=None,
    prior_logvar=None,
):
    r = reconstruction_loss(x, recon, kind=recon_kind, huber_delta=huber_delta)
    # free_bits applies to any KL-weighted ELBO, not just the β path. The
    # previous gate (``free_bits if objective == "beta" else 0.0``) effectively
    # disabled free-bits for the vanilla objective, which removed the only
    # safeguard against posterior collapse when β is fixed at 1 from epoch 0.
    kl_ps = kl_per_sample(
        mu,
        logvar,
        free_bits=(free_bits if objective in ("beta", "vanilla") else 0.0),
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
    )
    kl_mean = kl_ps.mean()
    if objective == "beta":
        loss = r + beta * kl_mean
    elif objective == "vanilla":
        # Route vanilla through the β scheduler too. ``beta`` is produced by
        # ``beta_schedule`` in the training loop and reaches ``beta_max`` (1.0
        # by default for vanilla) after the warm-up. Using the same path as
        # β-VAE lets the scheduler ramp KL pressure from 0 instead of hitting
        # the model with full unit weight at epoch 0.
        loss = r + beta * kl_mean
    elif objective == "capacity":
        loss = r + capacity_gamma * torch.abs(kl_ps - capacity_C).mean()
    else:
        raise ValueError("objective must be 'beta', 'vanilla', or 'capacity'")
    return loss, r, kl_mean


# ---------------------------------------------------------------------------
# DS-VAE helpers (cyclical β annealing, class-conditional KL, focal CE, SupCon)
# ---------------------------------------------------------------------------


def cyclical_beta_schedule(
    epoch: int,
    *,
    n_cycles: int = 4,
    cycle_len: int = 50,
    beta_max: float = 1.0,
    ramp_frac: float = 0.5,
) -> float:
    """Cyclical β annealing (Fu et al. 2019).

    Produces ``n_cycles`` linear 0 → ``beta_max`` ramps of length
    ``cycle_len`` epochs each. Within a cycle, β ramps linearly during the
    first ``ramp_frac`` of the cycle and stays at ``beta_max`` for the
    remainder. After ``n_cycles * cycle_len`` epochs β is pinned at
    ``beta_max`` indefinitely.

    Epoch numbering is 1-based to match the rest of the training loop
    (``beta_schedule``/``capacity_schedule``).
    """
    if beta_max <= 0.0 or cycle_len <= 0 or n_cycles <= 0:
        return float(beta_max)
    # Switch to 0-based for the cycle arithmetic.
    e = max(0, int(epoch) - 1)
    total = int(n_cycles) * int(cycle_len)
    if e >= total:
        return float(beta_max)
    pos = e % int(cycle_len)
    ramp_end = max(1, int(round(ramp_frac * cycle_len)))
    if pos >= ramp_end:
        return float(beta_max)
    return float(beta_max * pos / ramp_end)


def gaussian_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    *,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """Closed-form KL(N(μ_q, σ_q²) ‖ N(μ_p, σ_p²)) per-sample.

    Returns a tensor of shape ``(batch,)``. ``free_bits`` applies a per-dim
    floor *before* summing (same semantics as :func:`kl_per_sample`).
    """
    logvar_q = logvar_q.clamp(min=-30.0, max=20.0)
    logvar_p = logvar_p.clamp(min=-30.0, max=20.0)
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    diff = mu_q - mu_p
    per_dim = 0.5 * (
        (var_q + diff.pow(2)) / var_p - 1.0 + logvar_p - logvar_q
    )
    if free_bits > 0:
        per_dim = torch.clamp(per_dim, min=float(free_bits))
    return per_dim.sum(dim=-1)


def effective_number_class_weights(
    class_counts: torch.Tensor,
    *,
    beta: float = 0.9999,
) -> torch.Tensor:
    """Effective-number class weights (Cui et al. 2019).

    ``w_k = (1 - β) / (1 - β^{n_k})`` normalised so that
    ``sum(w_k) == n_classes``.
    """
    counts = class_counts.float().clamp(min=1.0)
    beta = float(beta)
    eff_num = 1.0 - torch.pow(torch.tensor(beta, dtype=counts.dtype), counts)
    weights = (1.0 - beta) / eff_num.clamp(min=1e-12)
    weights = weights * (counts.numel() / weights.sum().clamp(min=1e-12))
    return weights


def focal_ce_balanced(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    class_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Multi-class focal cross-entropy with class weights.

    ``loss_i = -w_{y_i} · (1 - p_{y_i})^γ · log p_{y_i}``.

    Handles soft labels too: when ``target`` has shape ``(batch, n_classes)``
    the weighted sum over classes is taken (useful for MixUp). Returns a
    scalar (mean over batch).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp().clamp(min=eps, max=1.0 - eps)

    if target.dim() == 1:
        # Hard labels.
        t = target.long()
        log_pt = log_probs.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        pt = probs.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        focal = (1.0 - pt).pow(float(gamma)) * (-log_pt)
        if class_weight is not None:
            w = class_weight.to(logits.device).gather(0, t)
            focal = focal * w
        return focal.mean()

    # Soft labels (e.g. from MixUp): ``target`` is a probability vector.
    focal = (1.0 - probs).pow(float(gamma)) * (-log_probs)
    if class_weight is not None:
        w = class_weight.to(logits.device).view(1, -1)
        focal = focal * w
    per_sample = (target * focal).sum(dim=-1)
    return per_sample.mean()


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Supervised Contrastive loss (Khosla et al. 2020).

    ``features`` should be ℓ₂-normalised.  Samples with a single instance
    of their class in the batch contribute nothing (no positive pair); if
    every class is singleton the loss returns ``0``.  When ``labels`` is
    one-hot / soft (MixUp), hard labels are taken via argmax.
    """
    if features.ndim != 2:
        raise ValueError("features must be shape (batch, dim)")
    if labels.dim() > 1:
        labels = labels.argmax(dim=-1)
    labels = labels.view(-1, 1)
    batch = features.size(0)
    if batch < 2:
        return features.new_zeros(())

    sim = torch.matmul(features, features.t()) / float(temperature)
    # Numerical stability: subtract per-row max, then zero-out the diagonal
    # via an additive mask.
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    logits_mask = 1.0 - torch.eye(batch, device=features.device)
    exp_sim = torch.exp(sim) * logits_mask

    mask_pos = (labels == labels.t()).float() * logits_mask  # same-class, off-diag

    denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=eps)
    log_prob = sim - torch.log(denom)
    pos_counts = mask_pos.sum(dim=1).clamp(min=eps)
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / pos_counts

    # Samples that have no positive pair are excluded from the mean.
    has_pos = (mask_pos.sum(dim=1) > 0).float()
    if has_pos.sum() < 1.0:
        return features.new_zeros(())
    return -(mean_log_prob_pos * has_pos).sum() / has_pos.sum()
