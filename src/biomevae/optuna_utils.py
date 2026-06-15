from __future__ import annotations

from typing import Any, Dict, Optional
import copy
import json
from pathlib import Path

__all__ = [
    "suggest_params",
    "build_trial_params",
    "load_search_space",
    "filter_trial_params",
]


def _fget(frozen: Dict[str, Any], trial, key: str, method: str, *args, **kwargs):
    """Get *key* from *frozen* if present, else call ``trial.<method>(key, ...)``."""
    if key in frozen:
        return frozen[key]
    return getattr(trial, method)(key, *args, **kwargs)


def suggest_params(
    trial,
    base: Dict[str, Any],
    *,
    frozen: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    frozen = frozen or {}
    params: Dict[str, Any] = {
        "device": base["device"],
        "epochs": base["epochs"],
        "val_split": base["val_split"],
        "early_stop": base["early_stop"],
        "model_type": base.get("model_type", "euclid"),
        "model_kwargs": copy.deepcopy(base.get("model_kwargs", {})),
        "tax_levels": base.get("tax_levels", []),
        "tax_loss_weight": base.get("tax_loss_weight", 0.0),
        "lap_weight": base.get("lap_weight", 0.0),
    }

    latent_dim = _fget(
        frozen, trial, "latent_dim", "suggest_categorical",
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24],
    )
    n_layers = _fget(frozen, trial, "n_layers", "suggest_int", 1, 10)
    hidden_choices = [128, 160, 192, 256, 320, 384, 512]
    hidden = [
        _fget(frozen, trial, f"h{i+1}", "suggest_categorical", hidden_choices)
        for i in range(n_layers)
    ]

    activation = _fget(frozen, trial, "activation", "suggest_categorical", ["leakyrelu", "gelu", "relu"])
    layer_norm = _fget(frozen, trial, "layer_norm", "suggest_categorical", [False, True])
    dropout = _fget(frozen, trial, "dropout", "suggest_float", 0.0, 0.2)

    optimizer = _fget(frozen, trial, "optimizer", "suggest_categorical", ["adam", "adamw"])
    weight_decay = (
        0.0
        if optimizer == "adam"
        else _fget(frozen, trial, "weight_decay", "suggest_float", 1e-5, 1e-3, log=True)
    )
    lr = _fget(frozen, trial, "lr", "suggest_float", 1e-4, 5e-3, log=True)
    batch_size = _fget(frozen, trial, "batch_size", "suggest_categorical", [32, 64, 128])
    grad_clip = _fget(frozen, trial, "grad_clip", "suggest_float", 0.5, 2.0)

    log1p = _fget(frozen, trial, "log1p", "suggest_categorical", [False, True])
    standardize = _fget(frozen, trial, "standardize", "suggest_categorical", [False, True])

    objective = _fget(frozen, trial, "objective", "suggest_categorical", ["beta", "vanilla", "capacity"])
    recon = _fget(frozen, trial, "recon", "suggest_categorical", ["mae", "huber", "mse"])
    huber_delta = (
        1.0
        if recon != "huber"
        else _fget(frozen, trial, "huber_delta", "suggest_float", 0.5, 2.0)
    )

    if objective == "vanilla":
        kl_warmup = 0
        kl_warmup_frac = None
        beta_max = 1.0
        free_bits = 0.0
    else:
        # Absolute ``kl_warmup`` is the historical default and remains the
        # canonical search dimension — a long warmup (e.g. 200-400 epochs
        # over 100 training epochs) is ML-effective for β-VAEs because it
        # keeps the KL pressure gentle while the decoder learns the latent
        # code, avoiding posterior collapse. ``kl_warmup_frac`` is honoured
        # when explicitly set (e.g. via a custom search space) but is not
        # sampled by default.
        kl_warmup = _fget(frozen, trial, "kl_warmup", "suggest_int", 200, 400)
        kl_warmup_frac = frozen.get("kl_warmup_frac")
        beta_max = _fget(frozen, trial, "beta_max", "suggest_float", 0.02, 0.3)
        free_bits = _fget(frozen, trial, "free_bits", "suggest_float", 0.0, 0.05) if objective == "beta" else 0.0

    capacity_start = (
        0.0
        if objective != "capacity"
        else _fget(frozen, trial, "capacity_start", "suggest_float", 0.0, 0.25 * latent_dim)
    )
    capacity_end = None
    capacity_epochs = 100
    capacity_gamma = 1.0
    if objective == "capacity":
        capacity_end = _fget(frozen, trial, "capacity_end", "suggest_float", 0.25 * latent_dim, 1.0 * latent_dim)
        capacity_epochs = _fget(frozen, trial, "capacity_epochs", "suggest_int", 50, 200)
        capacity_gamma = _fget(frozen, trial, "capacity_gamma", "suggest_float", 0.5, 4.0)

    if params["model_type"] == "hyperbolic":
        params["model_kwargs"]["curvature"] = _fget(frozen, trial, "curvature", "suggest_float", 0.25, 4.0, log=True)

    params.update(
        {
            "latent_dim": latent_dim,
            "hidden": hidden,
            "activation": activation,
            "layer_norm": layer_norm,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "log1p": log1p,
            "standardize": standardize,
            "objective": objective,
            "recon": recon,
            "huber_delta": huber_delta,
            "kl_warmup": kl_warmup,
            "kl_warmup_frac": kl_warmup_frac,
            "beta_max": beta_max,
            "free_bits": free_bits,
            "capacity_start": capacity_start,
            "capacity_end": capacity_end if objective == "capacity" else None,
            "capacity_epochs": capacity_epochs,
            "capacity_gamma": capacity_gamma,
        }
    )
    return params


def _assign_nested(params: Dict[str, Any], key: str, value: Any) -> None:
    if "." not in key:
        params[key] = value
        return
    current = params
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def load_search_space(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Optuna search space file not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Optuna search space configuration must be a JSON object.")
    return data


def _suggest_from_config(trial, config: Dict[str, Any]) -> Dict[str, Any]:
    suggested: Dict[str, Any] = {}
    for key, spec in config.items():
        if isinstance(spec, dict) and "method" in spec:
            method = spec["method"]
            kwargs = {k: v for k, v in spec.items() if k != "method"}
            if not hasattr(trial, method):
                raise ValueError(f"Optuna Trial has no method '{method}' for key '{key}'.")
            # Convert unhashable list choices to tuples for Optuna's
            # suggest_categorical, which requires hashable values.
            has_list_choices = (
                method == "suggest_categorical"
                and "choices" in kwargs
                and any(isinstance(c, list) for c in kwargs["choices"])
            )
            if has_list_choices:
                kwargs["choices"] = [
                    tuple(c) if isinstance(c, list) else c for c in kwargs["choices"]
                ]
            suggest_fn = getattr(trial, method)
            value = suggest_fn(key, **kwargs)
            suggested[key] = list(value) if has_list_choices and isinstance(value, tuple) else value
        else:
            suggested[key] = spec
    return suggested


def build_trial_params(
    trial,
    base: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = copy.deepcopy(base)
    suggested: Dict[str, Any] = _suggest_from_config(trial, config) if config else {}
    frozen: Dict[str, Any] = {}
    if config:
        for key, spec in config.items():
            if "." in key:
                continue
            if isinstance(spec, dict) and "method" in spec:
                # Use the already-suggested value so suggest_params can make
                # dependent decisions (e.g. optimizer -> weight_decay)
                # without seeing placeholder sentinels.
                frozen[key] = suggested[key]
                continue
            frozen[key] = spec
    params.update(suggest_params(trial, base, frozen=frozen))
    if suggested:
        for key, value in suggested.items():
            _assign_nested(params, key, value)
    return params


def filter_trial_params(params: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {k: v for k, v in params.items() if k not in ("tax_As", "lap_L", "feature_clades")}
    if not filtered.get("model_kwargs"):
        filtered.pop("model_kwargs", None)

    objective = filtered.get("objective")
    recon = filtered.get("recon")

    if objective != "beta":
        filtered.pop("free_bits", None)
    if objective in {"vanilla", "capacity"}:
        filtered.pop("kl_warmup", None)
        filtered.pop("kl_warmup_frac", None)
        filtered.pop("beta_max", None)
    if objective != "capacity":
        for key in ("capacity_start", "capacity_end", "capacity_epochs", "capacity_gamma"):
            filtered.pop(key, None)

    if recon != "huber":
        filtered.pop("huber_delta", None)

    tax_levels = filtered.get("tax_levels") or []
    tax_loss_weight = float(filtered.get("tax_loss_weight", 0.0) or 0.0)
    lap_weight = float(filtered.get("lap_weight", 0.0) or 0.0)
    if not tax_levels and tax_loss_weight <= 0.0 and lap_weight <= 0.0:
        for key in ("tax_levels", "tax_loss_weight", "lap_weight"):
            filtered.pop(key, None)

    return filtered
