"""Training CLI for DS-VAE (Disease-Supervised Phylogenetic VAE).

Usage::

    # Unsupervised variant (targets beating NMF on disease classification).
    biomevae-train-dsvae --input sgb_table.tsv --taxonomy phyla.tsv \\
        --outdir out/dsvae-unsup --no-supervised

    # Supervised variant (targets beating raw-count XGBoost).
    biomevae-train-dsvae --input sgb_table.tsv --taxonomy phyla.tsv \\
        --outdir out/dsvae-sup --supervised \\
        --metadata sample_metadata.tsv --label-col disease

The unsupervised variant optimises ``NB_NLL + β·KL`` with cyclical β
annealing and free-bits. The supervised variant additionally learns a
class-conditional prior, a focal-loss classifier head and a supervised
contrastive loss on ``μ_z`` — see :mod:`biomevae.models.dsvae` for the
full design.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from biomevae.data import load_matrix
from biomevae.models.tree_spec import build_tree_spec
from biomevae.optuna_utils import _suggest_from_config, load_search_space
from biomevae.taxonomy import load_feature_clades
from biomevae.trainers.train_loop import train_once_dsvae


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("biomevae-train-dsvae")
    ap.add_argument("--input", required=True)
    ap.add_argument("--taxonomy", required=True, help="Path to phyla.tsv/CSV.")
    ap.add_argument("--outdir", required=True)

    # Variant selector
    ap.add_argument(
        "--supervised",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the supervised variant (class-conditional prior, focal "
            "classifier head, SupCon loss). Requires --metadata/--label-col."
        ),
    )
    ap.add_argument(
        "--metadata",
        default=None,
        help=(
            "Sample-metadata TSV/CSV with per-sample labels. Required when "
            "--supervised."
        ),
    )
    ap.add_argument(
        "--label-col", default="disease",
        help="Metadata column holding the class label (default: disease).",
    )

    # Architecture
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument(
        "--hidden", nargs="+", type=int, default=[512, 256, 128],
    )
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pseudocount", type=float, default=0.5)
    ap.add_argument("--classifier-hidden", type=int, default=128)
    ap.add_argument(
        "--branchlen-mode", choices=["unit", "rank"], default="unit",
    )

    # Optimisation
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Cyclical β schedule
    ap.add_argument("--beta-max", type=float, default=1.0)
    ap.add_argument("--beta-n-cycles", type=int, default=4)
    ap.add_argument("--beta-cycle-len", type=int, default=50)
    ap.add_argument("--beta-ramp-frac", type=float, default=0.5)
    ap.add_argument("--free-bits", type=float, default=0.03)

    # Supervised-only knobs
    ap.add_argument("--gamma-cls", type=float, default=1.0,
                    help="Weight of the focal-CE classifier loss.")
    ap.add_argument("--gamma-con", type=float, default=0.3,
                    help="Weight of the supervised contrastive loss.")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--supcon-tau", type=float, default=0.1)
    ap.add_argument("--mixup-alpha", type=float, default=0.2,
                    help="Beta(α, α) for PhILR-space MixUp; 0 disables MixUp.")
    ap.add_argument("--effnum-beta", type=float, default=0.9999,
                    help="β for effective-number class weights (Cui 2019).")

    # Split / early stop
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Optuna
    ap.add_argument("--optuna", action="store_true")
    ap.add_argument("--optuna-trials", type=int, default=30)
    ap.add_argument("--optuna-config", type=str, default=None)
    return ap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_base_params(args) -> Dict[str, Any]:
    return {
        "device": args.device,
        "model_type": "dsvae",
        "supervised": bool(args.supervised),
        "latent_dim": int(args.latent_dim),
        "hidden": list(args.hidden),
        "dropout": float(args.dropout),
        "pseudocount": float(args.pseudocount),
        "classifier_hidden": int(args.classifier_hidden),
        "branchlen_mode": args.branchlen_mode,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "beta_max": float(args.beta_max),
        "beta_n_cycles": int(args.beta_n_cycles),
        "beta_cycle_len": int(args.beta_cycle_len),
        "beta_ramp_frac": float(args.beta_ramp_frac),
        "free_bits": float(args.free_bits),
        "gamma_cls": float(args.gamma_cls),
        "gamma_con": float(args.gamma_con),
        "focal_gamma": float(args.focal_gamma),
        "supcon_tau": float(args.supcon_tau),
        "mixup_alpha": float(args.mixup_alpha),
        "effnum_beta": float(args.effnum_beta),
        "val_split": float(args.val_split),
        "early_stop": int(args.early_stop),
    }


def _read_metadata_labels(
    metadata_path: str,
    label_col: str,
    sample_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Load per-sample labels from a metadata TSV/CSV.

    Returns an integer label array aligned to ``sample_names`` and the
    list of class names (index → class). Missing samples raise SystemExit
    with a clear error message.
    """
    path = str(metadata_path)
    if path.endswith((".tsv", ".txt", ".tab")):
        meta = pd.read_csv(path, sep="\t")
    else:
        # Try tab first, fall back to comma.
        try:
            meta = pd.read_csv(path, sep="\t")
            if meta.shape[1] <= 1:
                meta = pd.read_csv(path)
        except Exception:
            meta = pd.read_csv(path)
    if label_col not in meta.columns:
        raise SystemExit(
            f"biomevae-train-dsvae: label column '{label_col}' not in "
            f"metadata columns: {list(meta.columns)}"
        )

    # Identify the sample-id column. Common names vary across studies,
    # so we fall back to the first column if none of the usual names are
    # present.
    id_candidates = [
        "sample_id", "sample", "Sample", "Run", "run_accession",
        "sampleID", "ID", "id",
    ]
    id_col = None
    for c in id_candidates:
        if c in meta.columns:
            id_col = c
            break
    if id_col is None:
        id_col = meta.columns[0]
    meta = meta.set_index(id_col)

    missing = [s for s in sample_names if s not in meta.index]
    if missing:
        raise SystemExit(
            f"biomevae-train-dsvae: {len(missing)} samples are missing from "
            f"the metadata (first few: {missing[:5]})."
        )

    raw = meta.loc[list(sample_names), label_col].astype(str).fillna("NA")
    classes = sorted(raw.unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = raw.map(class_to_idx).to_numpy(dtype=np.int64)
    return y, classes


def _resolve_tree_spec(feature_clades, taxonomy_path, branchlen_mode):
    return build_tree_spec(feature_clades, taxonomy_path, branchlen_mode=branchlen_mode)


# ---------------------------------------------------------------------------
# Train / embed / save
# ---------------------------------------------------------------------------


def _finalise_run(
    args,
    params: Dict[str, Any],
    X_raw: np.ndarray,
    sample_names: List[str],
    feature_clades: List[str],
    labels: np.ndarray | None,
    label_names: List[str],
    res: Dict[str, Any],
) -> None:
    """Write embeddings.tsv, recon.tsv, config.json for the final model."""
    outdir = args.outdir
    model = res.get("model")
    if model is None:
        # Training failed or early stopped without saving; rebuild weights
        # from disk to get a deterministic artefact.
        from biomevae.models.dsvae import DSVAE

        tree_spec = _resolve_tree_spec(
            feature_clades, args.taxonomy, params["branchlen_mode"],
        )
        n_classes = len(label_names) if params["supervised"] else None
        model = DSVAE(
            n_features=X_raw.shape[1],
            latent_dim=int(params["latent_dim"]),
            tree_spec=tree_spec,
            supervised=bool(params["supervised"]),
            n_classes=n_classes,
            hidden=list(params["hidden"]),
            dropout=float(params["dropout"]),
            pseudocount=float(params["pseudocount"]),
            classifier_hidden=int(params["classifier_hidden"]),
        )
        model_path = os.path.join(outdir, "model.pt")
        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_raw.astype(np.float32)).to(device)
        mu_z, _ = model.encode(xt)
        lib = xt.sum(dim=1, keepdim=True).clamp(min=1.0)
        recon = model.decode(mu_z, lib).cpu().numpy()
        emb = mu_z.cpu().numpy()

    pd.DataFrame(
        emb, index=sample_names,
        columns=[f"z{i}" for i in range(emb.shape[1])],
    ).to_csv(os.path.join(outdir, "embeddings.tsv"), sep="\t")
    pd.DataFrame(
        recon, index=sample_names,
        columns=feature_clades if len(feature_clades) == recon.shape[1]
        else [f"f{i}" for i in range(recon.shape[1])],
    ).to_csv(os.path.join(outdir, "recon.tsv"), sep="\t")

    # Compose and persist config.json.
    tree_spec = _resolve_tree_spec(
        feature_clades, args.taxonomy, params["branchlen_mode"],
    )
    cfg: Dict[str, Any] = {k: v for k, v in params.items()}
    cfg.update({
        "model_type": "dsvae",
        "feature_clades": feature_clades,
        "tree_spec": tree_spec.to_json(),
        "n_samples": int(X_raw.shape[0]),
        "input_dim": int(X_raw.shape[1]),
        "activation": "silu",
        "layer_norm": True,
    })
    if params.get("supervised"):
        cfg["label_col"] = args.label_col
        cfg["class_names"] = list(label_names)
        cfg["n_classes"] = len(label_names)
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)


# ---------------------------------------------------------------------------
# Optuna runner
# ---------------------------------------------------------------------------


def _run_optuna(
    args,
    X_raw: np.ndarray,
    sample_names: List[str],
    feature_clades: List[str],
    labels: np.ndarray | None,
    label_names: List[str],
) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna not installed. `pip install biomevae[optuna]`."
        ) from exc

    config = None
    if args.optuna_config:
        config = load_search_space(args.optuna_config)

    study_dir = os.path.join(args.outdir, "optuna_trials")
    os.makedirs(study_dir, exist_ok=True)

    base_params = _prepare_base_params(args)
    base_params["feature_clades"] = list(feature_clades)
    base_params["taxonomy_path"] = args.taxonomy

    direction = "maximize" if base_params["supervised"] else "minimize"

    def objective(trial):
        # DS-VAE has its own hyperparameter space (cyclical β, gamma_cls,
        # mixup_alpha, ...) that does not overlap with the classic
        # suggest_params() knobs (objective, kl_warmup, ...).  Build the
        # trial params by overlaying only the user-provided search-space
        # entries so DS-VAE defaults are preserved for everything else.
        params = copy.deepcopy(base_params)
        if config:
            suggested = _suggest_from_config(trial, config)
            for key, value in suggested.items():
                params[key] = value
        trial_out = os.path.join(study_dir, f"trial_{trial.number:04d}")
        seed = args.seed + trial.number
        trial.set_user_attr("params", params)
        trial.set_user_attr("seed", seed)
        try:
            res = train_once_dsvae(
                X_raw, sample_names, trial_out, params,
                seed=seed, verbose=False, return_model=False,
                labels=labels,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float("-inf") if direction == "maximize" else float("inf")

        if base_params["supervised"]:
            score = res.get("val_macro_f1")
            if score is None or not np.isfinite(score):
                return float("-inf")
            return float(score)
        score = res.get("best_val")
        if score is None or not np.isfinite(score):
            return float("inf")
        return float(score)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=args.optuna_trials)

    best = study.best_trial
    if not np.isfinite(best.value):
        raise SystemExit("No finite objective values found.")
    bp = best.user_attrs["params"]
    bs = best.user_attrs["seed"]
    res = train_once_dsvae(
        X_raw, sample_names, args.outdir, bp,
        seed=bs, verbose=True, return_model=True,
        labels=labels,
    )
    _finalise_run(args, bp, X_raw, sample_names, feature_clades, labels, label_names, res)

    with open(os.path.join(args.outdir, "optuna_best_params.json"), "w") as fh:
        json.dump(bp, fh, indent=2)
    try:
        study.trials_dataframe().to_csv(
            os.path.join(args.outdir, "optuna_trials.csv"), index=False,
        )
    except Exception:  # pragma: no cover - best-effort artefact
        pass
    tag = "val_macro_f1" if base_params["supervised"] else "best_val"
    print(f"\nOptuna best #{best.number} | {tag}={best.value:.6f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X_raw, sample_names = load_matrix(args.input, log1p=False)
    feature_clades = load_feature_clades(args.input)

    labels: np.ndarray | None = None
    label_names: List[str] = []
    if args.supervised:
        if args.metadata is None:
            raise SystemExit(
                "biomevae-train-dsvae: --metadata is required when --supervised."
            )
        labels, label_names = _read_metadata_labels(
            args.metadata, args.label_col, sample_names,
        )
        if len(label_names) < 2:
            raise SystemExit(
                f"biomevae-train-dsvae: supervised mode needs >= 2 classes "
                f"(found: {label_names})."
            )

    if args.optuna:
        _run_optuna(
            args, X_raw, sample_names, feature_clades, labels, label_names,
        )
        return

    params = _prepare_base_params(args)
    params["feature_clades"] = list(feature_clades)
    params["taxonomy_path"] = args.taxonomy
    if args.supervised:
        params["n_classes"] = len(label_names)

    res = train_once_dsvae(
        X_raw, sample_names, args.outdir, params,
        seed=args.seed, verbose=True, return_model=True,
        labels=labels,
    )
    _finalise_run(args, params, X_raw, sample_names, feature_clades, labels, label_names, res)

    if args.supervised:
        f1 = res.get("val_macro_f1", float("nan"))
        bacc = res.get("val_balanced_accuracy", float("nan"))
        print(
            f"\nDone. val_macro_f1={f1:.4f} val_balanced_accuracy={bacc:.4f} "
            f"val_recon={res.get('val_recon', float('nan')):.4f}"
        )
    else:
        print(f"\nDone. best_val={res.get('best_val', float('nan')):.6f}")


if __name__ == "__main__":
    main()
