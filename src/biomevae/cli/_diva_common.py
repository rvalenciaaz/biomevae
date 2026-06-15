"""Helpers shared by every DIVA training CLI.

Centralises the routine work — argparse defaults, dataset preparation
from a merged multi-study TSV, batch-tensor packaging, training loop,
plus Optuna hyperparameter search — so the per-backbone CLIs only have
to wire up the model-specific likelihood and reconstruction term.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from biomevae.losses import beta_schedule


__all__ = [
    "DIVADatasetTensors",
    "DIVATrainingResult",
    "DEFAULT_DIVA_SEARCH_SPACE",
    "StudyBalancedBatchSampler",
    "add_optuna_cli_args",
    "build_diva_dataset",
    "domain_class_encoders",
    "diva_train_loop",
    "encode_full_dataset",
    "run_diva_optuna",
    "save_diva_outputs",
    "split_train_val",
]


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


@dataclass
class DIVADatasetTensors:
    """Tensors and lookup tables used throughout DIVA training."""

    x_log: torch.Tensor                  # (n, n_features) log(counts + eps)
    x_raw: torch.Tensor                  # (n, n_features) raw counts
    domain: torch.Tensor                 # (n,) int64 study index
    klass: torch.Tensor                  # (n,) int64 class index, -1 = missing
    sample_ids: List[str]                # rows of every tensor
    domain_classes: List[str]            # study_name lookup
    class_classes: List[str]             # disease label lookup
    feature_clades: List[str]            # column order


def domain_class_encoders(
    metadata: pd.DataFrame,
    sample_ids: Sequence[str],
    *,
    study_col: str = "study_name",
    label_col: str = "disease",
    missing_class_value: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Encode (study, class) categoricals as int64 arrays.

    ``klass`` is set to ``-1`` for samples whose ``label_col`` value is
    ``missing_class_value`` or NaN, so :class:`~biomevae.models.diva.DIVALoss`
    can skip the cross-entropy term on those rows.
    """
    by_id = metadata.reindex(sample_ids)
    domain_str = by_id[study_col].astype(str).values
    label_str = by_id[label_col].astype(str).fillna(missing_class_value).values

    domain_classes = sorted(set(domain_str.tolist()))
    domain_idx = np.array(
        [domain_classes.index(d) for d in domain_str], dtype=np.int64,
    )

    klass_classes = sorted(
        set(label_str.tolist()) - {missing_class_value, "nan", "NaN"}
    )
    klass_lookup = {c: i for i, c in enumerate(klass_classes)}
    klass_idx = np.array(
        [klass_lookup.get(c, -1) for c in label_str], dtype=np.int64,
    )

    return domain_idx, klass_idx, domain_classes, klass_classes


def build_diva_dataset(
    sgb_table_path: os.PathLike | str,
    metadata_path: os.PathLike | str,
    *,
    label_col: str = "disease",
    study_col: str = "study_name",
    eps: float = 1.0,
) -> DIVADatasetTensors:
    """Load the merged multi-study dataset and pack tensors for DIVA training."""
    from biomevae.loso import load_merged

    X_raw, sample_ids, feature_clades, metadata = load_merged(
        sgb_table_path, metadata_path, study_col=study_col,
    )
    X_log = np.log(X_raw + eps).astype(np.float32)

    domain_idx, klass_idx, domain_classes, klass_classes = domain_class_encoders(
        metadata, sample_ids,
        study_col=study_col, label_col=label_col,
    )

    return DIVADatasetTensors(
        x_log=torch.from_numpy(X_log),
        x_raw=torch.from_numpy(X_raw),
        domain=torch.from_numpy(domain_idx),
        klass=torch.from_numpy(klass_idx),
        sample_ids=list(sample_ids),
        domain_classes=domain_classes,
        class_classes=klass_classes,
        feature_clades=list(feature_clades),
    )


class StudyBalancedBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Sample batches that contain at least ``min_studies`` studies.

    Used by PhyloDIVA so the per-study CORAL term and the hierarchical
    domain critic are well-defined within every batch — both require
    ≥2 studies, the CORAL term additionally requires ≥2 samples per
    study to estimate covariances.

    Implementation: each epoch shuffles the per-study sample lists,
    then round-robins from each study, drawing ``per_study`` samples
    per turn until the underlying pool is exhausted.  Falls back to
    the i.i.d. shuffle when only one study is available.
    """

    def __init__(
        self,
        domain: torch.Tensor,
        batch_size: int,
        *,
        min_studies: int = 2,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        super().__init__(None)
        domain_arr = domain.cpu().numpy().astype(np.int64)
        self.batch_size = int(batch_size)
        self.min_studies = max(1, int(min_studies))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._study_indices: Dict[int, np.ndarray] = {}
        unique = np.unique(domain_arr)
        for s in unique:
            self._study_indices[int(s)] = np.where(domain_arr == s)[0]
        self._n = int(domain_arr.shape[0])
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)
        # Shuffle each study's index list.
        per_study = {
            s: rng.permutation(idx).tolist()
            for s, idx in self._study_indices.items()
        }
        if len(per_study) < self.min_studies:
            # Single-study or degenerate; behave like a plain shuffle.
            order = rng.permutation(self._n).tolist()
            for i in range(0, len(order), self.batch_size):
                batch = order[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch
            self._epoch += 1
            return

        # Round-robin: pull ceil(batch_size / n_studies) samples per
        # study until the global pool runs out.  Random permutation of
        # study order each batch keeps statistics symmetric.
        total = sum(len(v) for v in per_study.values())
        emitted = 0
        n_studies = len(per_study)
        per_study_quota = max(1, self.batch_size // n_studies)
        while emitted < total:
            order = list(per_study.keys())
            rng.shuffle(order)
            batch: List[int] = []
            for s in order:
                pool = per_study[s]
                k = min(per_study_quota, len(pool))
                batch.extend(pool[:k])
                per_study[s] = pool[k:]
                if len(batch) >= self.batch_size:
                    break
            if not batch:
                break  # all studies exhausted
            emitted += len(batch)
            if len(batch) < self.batch_size and self.drop_last and emitted >= total:
                # Drop the trailing partial batch only when nothing else
                # is coming; prevents tiny final batches that destabilise
                # CORAL covariance estimates.
                return
            # Truncate to exactly batch_size when possible.
            if len(batch) > self.batch_size:
                batch = batch[: self.batch_size]
            yield batch
        self._epoch += 1

    def __len__(self) -> int:
        if self.drop_last:
            return self._n // self.batch_size
        return (self._n + self.batch_size - 1) // self.batch_size


def split_train_val(
    n: int, val_frac: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1).")
    rng = np.random.RandomState(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    n_val = min(n_val, n - 1)
    return idx[n_val:], idx[:n_val]


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------


@dataclass
class DIVATrainingResult:
    log_rows: List[dict]
    best_val: float


def diva_train_loop(
    *,
    model: torch.nn.Module,
    forward_fn: Callable[..., dict],
    recon_nll: Callable[[dict], torch.Tensor],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    beta_max: float,
    kl_warmup_frac: float,
    free_bits: float,
    alpha_d: float,
    alpha_y: float,
    grad_clip: float,
    early_stop: int,
    outdir: Path,
    device: torch.device,
    diva_combine: Callable[..., torch.Tensor],
    verbose: bool = True,
    extra_loss_fn: Optional[Callable[..., Dict[str, torch.Tensor]]] = None,
    on_epoch_start: Optional[Callable[[float], None]] = None,
) -> DIVATrainingResult:
    """Training loop shared by every DIVA backbone.

    ``forward_fn(model, batch)`` runs the model's ``forward`` and returns
    a dict that includes a ``"diva"`` key with the
    :class:`~biomevae.models.diva.DIVALossOutputs` bundle.

    ``recon_nll(out)`` produces the per-batch reconstruction NLL (NB or
    ZINB) given the dict returned by ``forward_fn``.

    ``diva_combine(diva_out, beta, alpha_d, alpha_y, batch_size)``
    returns the weighted DIVA term (KLs + auxiliary CEs).  Both
    backbones expose this as a static method.

    ``extra_loss_fn(out, batch, epoch_t) -> dict[str, Tensor]`` is an
    optional callback that adds extra penalties (e.g. PhyloDIVA's
    hierarchical clade critic, BM smoothness, CORAL) on top of the DIVA
    objective.  Each value is added to the loss and recorded under its
    own key in the training log.  ``epoch_t = ep / epochs`` in ``[0, 1]``.

    ``on_epoch_start(epoch_t)`` is called once per epoch (before the
    train loop) so callbacks can update mutable schedules (e.g. the GRL
    coefficient on a hierarchical critic).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )
    warmup = max(1, int(epochs * kl_warmup_frac))
    log_rows: list[dict] = []
    best_val = float("inf")
    no_improve = 0
    model_path = outdir / "model.pt"

    extra_keys: List[str] = []  # populated on first call so logging is consistent

    def _step(batch, training: bool, *, epoch_t: float) -> Dict[str, float]:
        out = forward_fn(model, batch, free_bits=free_bits)
        nll = recon_nll(out)
        diva_term = diva_combine(
            out["diva"],
            beta=beta, alpha_d=alpha_d, alpha_y=alpha_y,
            batch_size=out["mu_y"].size(0),
        )
        loss = nll + diva_term

        extras: Dict[str, torch.Tensor] = {}
        if extra_loss_fn is not None:
            extras = extra_loss_fn(out, batch, epoch_t) or {}
            for v in extras.values():
                loss = loss + v
            for k in extras:
                if k not in extra_keys:
                    extra_keys.append(k)

        if training:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        row: Dict[str, float] = {
            "nll": float(nll.item()),
            "kl_d": float(out["diva"].kl_d.item()),
            "kl_y": float(out["diva"].kl_y.item()),
            "kl_x": float(out["diva"].kl_x.item()),
            "ce_d": float(out["diva"].ce_d.item()),
            "ce_y": float(out["diva"].ce_y.item()),
            "loss": float(loss.item()),
            "bsz": int(out["mu_y"].size(0)),
        }
        for k, v in extras.items():
            row[k] = float(v.detach().item())
        return row

    base_keys = ("nll", "kl_d", "kl_y", "kl_x", "ce_d", "ce_y", "loss")

    for ep in range(1, epochs + 1):
        beta = beta_schedule(ep, warmup, beta_max)
        epoch_t = float(ep) / float(max(1, epochs))
        if on_epoch_start is not None:
            on_epoch_start(epoch_t)

        # ----- train -----
        model.train()
        agg = {k: 0.0 for k in base_keys}
        for k in extra_keys:
            agg[k] = 0.0
        agg_n = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            r = _step(batch, training=True, epoch_t=epoch_t)
            for k in list(agg.keys()):
                agg[k] += r.get(k, 0.0) * r["bsz"]
            for k in extra_keys:
                if k not in agg:
                    agg[k] = r.get(k, 0.0) * r["bsz"]
            agg_n += r["bsz"]
        train_metrics = {k: v / max(1, agg_n) for k, v in agg.items()}

        # ----- val -----
        model.eval()
        val_agg = {k: 0.0 for k in agg}
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                r = _step(batch, training=False, epoch_t=epoch_t)
                for k in val_agg:
                    val_agg[k] += r.get(k, 0.0) * r["bsz"]
                val_n += r["bsz"]
        val_metrics = {k: v / max(1, val_n) for k, v in val_agg.items()}

        row = {
            "epoch": ep, "beta": float(beta),
            "train_loss": train_metrics["loss"],
            "train_nll": train_metrics["nll"],
            "train_recon": train_metrics["nll"],
            "train_kl_d": train_metrics["kl_d"],
            "train_kl_y": train_metrics["kl_y"],
            "train_kl_x": train_metrics["kl_x"],
            "train_ce_d": train_metrics["ce_d"],
            "train_ce_y": train_metrics["ce_y"],
            "val_loss": val_metrics["loss"],
            "val_nll": val_metrics["nll"],
            "val_recon": val_metrics["nll"],
            "val_kl_d": val_metrics["kl_d"],
            "val_kl_y": val_metrics["kl_y"],
            "val_kl_x": val_metrics["kl_x"],
            "val_ce_d": val_metrics["ce_d"],
            "val_ce_y": val_metrics["ce_y"],
        }
        for k in extra_keys:
            row[f"train_{k}"] = train_metrics.get(k, 0.0)
            row[f"val_{k}"] = val_metrics.get(k, 0.0)
        log_rows.append(row)
        if not np.isfinite(row["train_loss"]) or not np.isfinite(row["val_loss"]):
            if verbose:
                print("Stopping: non-finite loss.")
            break

        scheduler.step(row["val_nll"])

        improved = row["val_nll"] + 1e-9 < best_val
        if early_stop > 0:
            if improved:
                best_val = row["val_nll"]
                no_improve = 0
                torch.save(model.state_dict(), model_path)
            else:
                no_improve += 1
                if no_improve >= early_stop:
                    if verbose:
                        print("Early stopping.")
                    break
        else:
            if improved:
                best_val = row["val_nll"]
            torch.save(model.state_dict(), model_path)

        if verbose:
            print(
                f"ep {ep:03d} | β={beta:.3f} | "
                f"loss={row['train_loss']:.2f} (nll={row['train_nll']:.2f}, "
                f"kl_d={row['train_kl_d']:.2f} kl_y={row['train_kl_y']:.2f} "
                f"kl_x={row['train_kl_x']:.2f}) | "
                f"val={row['val_loss']:.2f} (nll={row['val_nll']:.2f}) | "
                f"ce_d={row['train_ce_d']:.2f} ce_y={row['train_ce_y']:.2f}"
            )

    if model_path.exists():
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    return DIVATrainingResult(log_rows=log_rows, best_val=best_val)


def _move_batch(batch, device: torch.device):
    return tuple(t.to(device, non_blocking=True) for t in batch)


# ---------------------------------------------------------------------------
# Embedding + persistence
# ---------------------------------------------------------------------------


def encode_full_dataset(
    *,
    model: torch.nn.Module,
    encode_fn: Callable[..., Dict[str, torch.Tensor]],
    inputs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run the encoder over a tensor batched by ``batch_size``.

    Returns a dict with keys ``mu_d``, ``mu_y``, ``mu_x`` and the
    concatenated ``mu`` (numpy arrays, in dataset row order).
    """
    model.eval()
    parts = {"mu_d": [], "mu_y": [], "mu_x": []}
    n = inputs.size(0)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = inputs[start : start + batch_size].to(device)
            enc = encode_fn(batch)
            for k in parts:
                parts[k].append(enc[k].cpu().numpy())
    out = {k: np.concatenate(parts[k], axis=0) for k in parts}
    out["mu"] = np.concatenate([out["mu_d"], out["mu_y"], out["mu_x"]], axis=1)
    return out


def save_diva_outputs(
    *,
    outdir: os.PathLike | str,
    sample_ids: Sequence[str],
    feature_clades: Sequence[str],
    embeddings: Dict[str, np.ndarray],
    recon: Optional[np.ndarray],
    log_rows: List[dict],
    config: Dict[str, Any],
) -> None:
    """Write the standard biomevae ``model/`` artefact set, plus DIVA-specific files.

    Standard artefacts:
        embeddings.tsv   — *full* latent (z_d ‖ z_y ‖ z_x) for backwards
                           compatibility with biomevae-classify.
        recon.tsv        — reconstruction (when ``recon`` is provided).
        training_log.tsv — per-epoch loss decomposition.
        config.json      — model_type, dims, plus the DIVA factor split.

    DIVA-specific artefacts:
        embeddings_z_y.tsv — class-anchored latent only.  Recommended
                             input to downstream classifiers.
        embeddings_z_x.tsv — residual latent (the most domain-invariant
                             slice, useful for cross-study transfer).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Full latent (kept under the conventional name for downstream tools).
    full = embeddings["mu"]
    pd.DataFrame(
        full, index=sample_ids,
        columns=[f"z{i}" for i in range(full.shape[1])],
    ).to_csv(outdir / "embeddings.tsv", sep="\t")

    # Per-factor slices.
    for factor in ("z_d", "z_y", "z_x"):
        key = f"mu_{factor.split('_')[1]}"
        arr = embeddings[key]
        pd.DataFrame(
            arr, index=sample_ids,
            columns=[f"{factor}{i}" for i in range(arr.shape[1])],
        ).to_csv(outdir / f"embeddings_{factor}.tsv", sep="\t")

    if recon is not None:
        pd.DataFrame(
            recon, index=sample_ids, columns=feature_clades,
        ).to_csv(outdir / "recon.tsv", sep="\t")

    pd.DataFrame(log_rows).to_csv(
        outdir / "training_log.tsv", sep="\t", index=False,
    )

    with (outdir / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)


# ---------------------------------------------------------------------------
# Optuna hyperparameter search (shared by every DIVA backbone)
# ---------------------------------------------------------------------------
#
# Brings DIVA training to parity with the single-study pipeline (see
# ``vae_train.py::_run_optuna``): N trials with different seeds, the best
# trial is retrained in the canonical outdir, and ``optuna_best_params.json``
# / ``optuna_trials.csv`` are written next to the saved model.  Used by
# ``vae_train_diva_hyp_philrvae`` and ``vae_train_diva_betavae``.

# Default search space — only the knobs every DIVA backbone exposes via
# argparse with the same attribute name.  Backbone-specific knobs (e.g.
# the diva-hyp-philrvae
# ``curvature``) can be overridden via ``--optuna-config``.  ``hidden``
# is intentionally excluded because the three DIVA CLIs disagree on its
# shape (int vs list); pass it explicitly via ``--optuna-config`` if you
# want to search it for one specific backbone.
DEFAULT_DIVA_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "latent_d": {"method": "suggest_categorical", "choices": [2, 4, 6, 8]},
    "latent_y": {"method": "suggest_categorical", "choices": [4, 8, 12, 16]},
    "latent_x": {"method": "suggest_categorical", "choices": [4, 8, 12, 16]},
    "lr": {"method": "suggest_float", "low": 1e-4, "high": 5e-3, "log": True},
    "dropout": {"method": "suggest_float", "low": 0.0, "high": 0.3},
    # DIVA paper recommends alpha_y in 10–100; widen lower to 1.0 so the
    # search can demote the auxiliary classifier when DA is over-strong
    # for a particular backbone.
    "alpha_y": {
        "method": "suggest_float",
        "low": 1.0,
        "high": 100.0,
        "log": True,
    },
    "alpha_d": {
        "method": "suggest_float",
        "low": 0.5,
        "high": 5.0,
        "log": True,
    },
    # The three DIVA CLIs disagree on a sensible β default
    # (NB-likelihood: 1.0, β-VAE: 0.05).  Cover both regimes with a
    # log-uniform sweep so neither side gets boxed out of its native
    # collapse-free range.
    "beta_max": {
        "method": "suggest_float",
        "low": 0.02,
        "high": 1.0,
        "log": True,
    },
    "kl_warmup_frac": {"method": "suggest_float", "low": 0.1, "high": 0.5},
    "free_bits": {"method": "suggest_float", "low": 0.0, "high": 0.05},
    "batch_size": {
        "method": "suggest_categorical",
        "choices": [32, 64, 128],
    },
}


def add_optuna_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register the standard ``--optuna`` flags on a DIVA training CLI.

    Mirrors ``biomevae-train-*`` (see ``vae_train.py``) so the LOSO and
    single-study pipelines can drive every backbone with the same
    ``extra_args`` string.
    """
    parser.add_argument(
        "--optuna",
        action="store_true",
        help=(
            "Run Optuna hyperparameter search instead of a single training "
            "run (matches biomevae-train --optuna)."
        ),
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=30,
        help="Number of Optuna trials when --optuna is enabled.",
    )
    parser.add_argument(
        "--optuna-config",
        type=str,
        default=None,
        help=(
            "Optional JSON file overriding the default DIVA search space "
            "(see configs/optuna_search_space_diva.template.json)."
        ),
    )


def _hashable_choices(choices: Sequence[Any]) -> List[Any]:
    """Convert nested-list choices to tuples (Optuna requires hashable)."""
    return [tuple(c) if isinstance(c, list) else c for c in choices]


def _suggest_one(trial, key: str, spec: Any) -> Any:
    """Sample a single hyperparameter from a search-space spec.

    ``spec`` is either:

      * a JSON object with a ``method`` key naming an Optuna ``suggest_*``
        function and the remaining keys forwarded as kwargs, OR
      * a literal value, returned as-is (a "frozen" hyperparameter).
    """
    if not (isinstance(spec, dict) and "method" in spec):
        return spec
    method = spec["method"]
    if not hasattr(trial, method):
        raise ValueError(
            f"Optuna Trial has no method '{method}' for key '{key}'."
        )
    kwargs = {k: v for k, v in spec.items() if k != "method"}
    has_list = (
        method == "suggest_categorical"
        and "choices" in kwargs
        and any(isinstance(c, list) for c in kwargs["choices"])
    )
    if has_list:
        kwargs["choices"] = _hashable_choices(kwargs["choices"])
    value = getattr(trial, method)(key, **kwargs)
    return list(value) if has_list and isinstance(value, tuple) else value


def _apply_overrides(
    args: argparse.Namespace,
    overrides: Dict[str, Any],
    *,
    warn_unknown: bool = True,
) -> None:
    """Set each ``key=value`` in *overrides* on the argparse namespace.

    Keys must match argparse attribute names (underscores, not hyphens).
    By default a key that does not already exist on ``args`` triggers a
    runtime warning — this catches typos like ``latent_dim`` (the
    single-study attribute name) being supplied where the DIVA CLIs
    expect ``latent_d`` / ``latent_y`` / ``latent_x``, which would
    otherwise silently no-op.  Unknown keys are still applied so callers
    can pass DIVA-only knobs to a non-DIVA backbone if they really mean
    to.
    """
    import warnings

    for key, value in overrides.items():
        if warn_unknown and not hasattr(args, key):
            warnings.warn(
                f"Optuna override '{key}' does not match any CLI argument; "
                "it will be set on the argparse namespace but the training "
                "function will likely ignore it.",
                stacklevel=2,
            )
        setattr(args, key, value)


def run_diva_optuna(
    args: argparse.Namespace,
    train_fn: Callable[..., Dict[str, Any]],
    *,
    direction: str = "minimize",
    default_search_space: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Drive an Optuna search for a DIVA training CLI.

    Parameters
    ----------
    args
        Parsed argparse namespace; mutated in-place between trials so
        ``train_fn`` only has to read attributes off it.
    train_fn
        ``train_fn(args, outdir, *, verbose=False) -> dict`` runs one full
        DIVA training pass and returns at least ``{"best_val": float}``.
    direction
        Optuna optimisation direction (default: minimise val NLL).
    default_search_space
        Search-space overrides applied when ``--optuna-config`` is not
        provided.  Defaults to :data:`DEFAULT_DIVA_SEARCH_SPACE`.

    Returns the result dict from the final retraining run.
    """
    if args.optuna_trials <= 0:
        raise SystemExit("--optuna-trials must be a positive integer.")
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise SystemExit(
            "Optuna is not installed. Install with `pip install biomevae[optuna]`."
        ) from exc

    if args.optuna_config:
        from biomevae.optuna_utils import load_search_space

        try:
            search_space = load_search_space(args.optuna_config)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    else:
        search_space = (
            default_search_space
            if default_search_space is not None
            else DEFAULT_DIVA_SEARCH_SPACE
        )

    base_args_state = copy.deepcopy(vars(args))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    study_dir = outdir / "optuna_trials"
    study_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        # Reset to the user-provided defaults each trial so the previous
        # trial's overrides don't leak in.
        for key, value in base_args_state.items():
            setattr(args, key, value)

        overrides = {
            key: _suggest_one(trial, key, spec)
            for key, spec in search_space.items()
        }
        _apply_overrides(args, overrides)

        seed = int(base_args_state.get("seed", 42)) + trial.number
        args.seed = seed

        trial_outdir = study_dir / f"trial_{trial.number:04d}"
        trial_outdir.mkdir(parents=True, exist_ok=True)
        try:
            res = train_fn(args, trial_outdir, verbose=False)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float("-inf") if direction == "maximize" else float("inf")

        trial.set_user_attr("overrides", overrides)
        trial.set_user_attr("seed", seed)
        score = float(res.get("best_val", float("inf")))
        if not np.isfinite(score):
            return float("-inf") if direction == "maximize" else float("inf")
        return score

    study = optuna.create_study(direction=direction)
    # ``catch=(Exception,)`` keeps the search going when a single trial
    # fails (numerical instability from an aggressive search-space
    # sample, a model-class precondition violation, etc.).  Optuna marks
    # the trial FAILED, stores the traceback, and proceeds — much
    # better than aborting the whole sweep on the first stumble.
    # ``KeyboardInterrupt`` derives from ``BaseException`` and is not
    # caught, so Ctrl-C still works.
    study.optimize(
        objective, n_trials=int(args.optuna_trials), catch=(Exception,),
    )

    try:
        best = study.best_trial
    except ValueError as exc:
        # Optuna raises if every trial errored out without recording a
        # value; surface that as a friendlier SystemExit so the
        # Snakemake log points at the actual root cause (CUDA OOM, bad
        # search space, missing data) rather than an opaque traceback.
        raise SystemExit(
            f"Optuna: every trial failed before recording a val loss "
            f"({exc}). Inspect the per-trial logs under "
            f"{study_dir} for the original error."
        ) from exc

    if not np.isfinite(best.value):
        raise SystemExit("Optuna: no finite val losses found across trials.")
    best_overrides = best.user_attrs["overrides"]
    best_seed = int(best.user_attrs["seed"])

    # Restore defaults, then apply the best trial's overrides for the
    # final retrain in the user-requested outdir.
    for key, value in base_args_state.items():
        setattr(args, key, value)
    _apply_overrides(args, best_overrides)
    args.seed = best_seed

    res = train_fn(args, outdir, verbose=True)

    serialisable = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in best_overrides.items()
    }
    serialisable["seed"] = best_seed
    with (outdir / "optuna_best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    try:
        study.trials_dataframe().to_csv(
            outdir / "optuna_trials.csv", index=False,
        )
    except Exception:  # pragma: no cover - best-effort artefact
        pass

    print(
        f"\nOptuna best trial #{best.number} | val={best.value:.6f} | "
        f"seed={best_seed}"
    )
    return res
