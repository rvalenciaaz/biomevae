# ──────────────────────────────────────────────────────────────────────
# common.smk – shared definitions for the single-study and meta pipelines
#
# This file is ``include:``d from both ``workflow/Snakefile`` (single-study
# entry point) and ``workflow/Snakefile.meta`` (meta entry point that runs
# the full workflow across every multi-label study in the
# extract-microbiome-data registry).
#
# It provides:
#   * the canonical list of biomevae model variants to train
#   * helpers to resolve per-study input/output paths
#   * ``resolve_studies()`` which turns the Snakemake ``config`` into a
#     concrete list of studies – either from an explicit ``study``/``studies``
#     entry or, when ``auto_multi_label: true``, from the curatedMetagenomicData
#     study registry shipped with ``extract-microbiome-data``.
# ──────────────────────────────────────────────────────────────────────

import os
import sys
from pathlib import Path


# ── required config ─────────────────────────────────────────────────
if "data_root" not in config:
    raise ValueError(
        "Snakemake config must set 'data_root' – the directory containing "
        "<study_name>/{sgb_table.tsv,phyla.tsv,sample_metadata.tsv}."
    )
if "output_root" not in config:
    raise ValueError(
        "Snakemake config must set 'output_root' – where per-study models, "
        "figures and aggregate results are written."
    )

DATA_ROOT = str(Path(config["data_root"]).expanduser().resolve())
OUTPUT_ROOT = str(Path(config["output_root"]).expanduser().resolve())
LABEL = config.get("label", "disease")
EXTRA_ARGS = config.get("extra_args", "--epochs 100 --optuna --optuna-trials 100")
STUDY_DISPLAY_NAMES = dict(config.get("study_display_names", {}))

# ── evaluation seeds ───────────────────────────────────────────────
# Every downstream evaluation (classification, reconstruction CV,
# enterosignature refresh) is repeated over these seeds and the
# per-seed fold metrics are pooled. Kept in sync with
# ``biomevae.classify.DEFAULT_EVAL_SEEDS``.
EVAL_SEEDS = [int(s) for s in config.get("eval_seeds", [42, 43, 44, 45, 46])]
EVAL_SEEDS_STR = " ".join(str(s) for s in EVAL_SEEDS)

# ── extract-microbiome-data registry import ─────────────────────────
# Allow pulling the study registry from a sibling checkout or an
# explicit path in config.
_DEFAULT_EXTRACT = str(Path(workflow.basedir).resolve().parents[1] / "extract-microbiome-data")
_EXTRACT_PATH = config.get("extract_microbiome_data_path", _DEFAULT_EXTRACT)
if _EXTRACT_PATH and _EXTRACT_PATH not in sys.path:
    sys.path.insert(0, _EXTRACT_PATH)


# ── model catalogue ─────────────────────────────────────────────────
# The keys are the directory names used under ``<study>/models/<key>/``
# and every value must point at a console script registered in
# ``pyproject.toml`` (``[project.scripts]``).  Adding or renaming an
# entry here is sufficient to wire the model into the train /
# postprocess / classify / figures / aggregate DAG.
#
# Stays in sync with ``hpc/submit_all.sh`` and the entry points declared
# in ``pyproject.toml``.  When a new training CLI lands in
# ``biomevae.cli`` it should be added here so the single-study and meta
# Snakemake pipelines pick it up automatically.
MODELS = {
    # ── Unsupervised baselines ─────────────────────────────────────
    "beta-vae":      {"cmd": "biomevae-train",           "needs_tax": False, "needs_metadata": False},
    "vanilla-vae":   {"cmd": "biomevae-train-vanilla",   "needs_tax": False, "needs_metadata": False},
    "hyp-vae":       {"cmd": "biomevae-train-hyp",       "needs_tax": False, "needs_metadata": False},
    "tax-vae":       {"cmd": "biomevae-train-tax",       "needs_tax": True,  "needs_metadata": False},
    "hyp-tax-vae":   {"cmd": "biomevae-train-hyp-tax",   "needs_tax": True,  "needs_metadata": False},
    "graph-vae":     {"cmd": "biomevae-train-graph",     "needs_tax": True,  "needs_metadata": False},
    "treeprior-vae": {"cmd": "biomevae-train-treeprior", "needs_tax": True,  "needs_metadata": False},
    "fuse-vae":      {"cmd": "biomevae-train-fuse",      "needs_tax": True,  "needs_metadata": False},

    # ── Tree-structured DTM VAEs ───────────────────────────────────
    # ``tree-dtm-vae`` is the plain backbone; the ``diva-`` /
    # ``phylodiva-`` / ``taxi-`` variants reuse the same likelihood
    # but layer domain-invariant or phylogeny-aware encoders on top.
    # The cross-domain wrappers need multiple studies acting as
    # ``domain`` labels, so they are flagged ``cross_study_only`` and
    # dropped from the catalogue in single-study mode (see the filter
    # block after the MODELS dict).
    "tree-dtm-vae":         {"cmd": "biomevae-train-tree-dtm",          "needs_tax": True,  "needs_metadata": False},
    "diva-tree-dtm-vae":    {"cmd": "biomevae-train-diva-tree-dtm",     "needs_tax": True,  "needs_metadata": True,  "cross_study_only": True},
    "phylodiva-tree-dtm-vae": {"cmd": "biomevae-train-phylodiva-tree-dtm", "needs_tax": True, "needs_metadata": True, "cross_study_only": True},
    "taxi-tree-dtm-vae":    {"cmd": "biomevae-train-taxi-tree-dtm",     "needs_tax": True,  "needs_metadata": True,  "cross_study_only": True},

    # ── PhILR-NB family ────────────────────────────────────────────
    # Both variants share the PhILR-NB compositional backbone and
    # only differ in the latent geometry: Euclidean (``philrvae``)
    # versus a Poincaré-ball latent with tangent-space Gaussian and
    # expmap0 reparam (``hyp-philrvae``). The hyperbolic variant
    # requires the ``hyper`` extra (geoopt).
    #
    # The previous catalogue exposed a third key ``hyp-philr-zinb``
    # that was a hard duplicate of ``hyp-philrvae`` (same CLI, no
    # distinguishing flag and no ZINB likelihood implemented in the
    # PhILR backbone), so it has been removed.
    "philrvae":     {"cmd": "biomevae-train-philrvae",     "needs_tax": True, "needs_metadata": False},
    "hyp-philrvae": {"cmd": "biomevae-train-hyp-philrvae", "needs_tax": True, "needs_metadata": False},

    # ── DIVA / PhyloDIVA wrappers (need disease labels) ────────────
    # These are cross-domain wrappers: the encoder learns to be
    # invariant to a ``domain`` factor that is meaningful only when
    # there are multiple cohorts in the training set. They are flagged
    # ``cross_study_only`` so the single-study pipeline skips them.
    "diva-beta-vae":            {"cmd": "biomevae-train-diva-beta-vae",          "needs_tax": False, "needs_metadata": True, "cross_study_only": True},
    "diva-hyp-philrvae":        {"cmd": "biomevae-train-diva-hyp-philrvae",      "needs_tax": True,  "needs_metadata": True, "cross_study_only": True},
    "phylodiva-beta-vae":       {"cmd": "biomevae-train-phylodiva-beta-vae",     "needs_tax": False, "needs_metadata": True, "cross_study_only": True},
    "phylodiva-hyp-philrvae":   {"cmd": "biomevae-train-phylodiva-hyp-philrvae", "needs_tax": True,  "needs_metadata": True, "cross_study_only": True},
    "taxi-hyp-philrvae":        {"cmd": "biomevae-train-taxi-hyp-philrvae",      "needs_tax": True,  "needs_metadata": True, "cross_study_only": True},

    # NOTE: ``flowxformer-vae`` and ``hgvae-zi`` have been removed from
    # the workflow catalogue: their training CLIs are deprecated and
    # are not exercised by any current pipeline (also unsupported by
    # ``hpc/postprocess_model.sh``'s interpret/test stages). The Python
    # implementations remain importable in ``biomevae.models`` for any
    # downstream user that needs them, but they no longer run as part
    # of the Snakemake DAG.

    # ── DS-VAE variants (disease-supervised PhILR-NB) ──────────────
    # ``dsvae-unsup`` trains the stock PhILR-NB backbone with cyclical
    # β annealing + free-bits; ``dsvae-sup`` adds a class-conditional
    # prior and focal/SupCon classifier head and therefore needs the
    # sample-metadata table.  See ``src/biomevae/models/dsvae.py``.
    "dsvae-unsup": {
        "cmd": "biomevae-train-dsvae --no-supervised",
        "needs_tax": True,
        "needs_metadata": False,
    },
    "dsvae-sup": {
        "cmd": "biomevae-train-dsvae --supervised",
        "needs_tax": True,
        "needs_metadata": True,
    },

    # ── CAPDA-VAE (single-study) ───────────────────────────────────
    # Single-cohort sibling of the LOSO ``capda-vae``: multi-resolution
    # CLR taxonomy bias + a supervised VAE whose class-head probabilities
    # are produced leak-free out-of-fold (stratified K-fold) and stacked
    # with the log1p species features for the downstream classifier.  The
    # cross-study conditional alignment is inert with one cohort, so —
    # unlike the diva-/phylodiva-/taxi- wrappers — this variant is NOT
    # ``cross_study_only`` and stays in the single-study catalogue.
    "capda-vae": {
        "cmd": "biomevae-train-capda-vae-ss",
        "needs_tax": True,
        "needs_metadata": True,
    },
}
# Ensure every entry exposes ``needs_metadata`` / ``cross_study_only`` —
# downstream rules key off these flags to decide whether to pass
# ``--metadata`` and whether to include the model at all.
for _model_name, _spec in MODELS.items():
    _spec.setdefault("needs_metadata", False)
    _spec.setdefault("cross_study_only", False)

# ── single-study filter ────────────────────────────────────────────
# ``workflow/Snakefile`` (the single-study entry point) sets
# ``single_study_mode: true`` before including this file; the meta /
# loso entry points leave it false.  Cross-domain wrappers
# (``diva-*`` / ``phylodiva-*`` / ``taxi-*``) require multiple cohorts
# to act as ``domain`` labels, so they are dropped from the catalogue
# in single-study mode rather than silently training a degenerate
# single-domain encoder.
if config.get("single_study_mode", False):
    _dropped = [name for name, spec in MODELS.items() if spec.get("cross_study_only")]
    for _name in _dropped:
        MODELS.pop(_name)
    if _dropped:
        sys.stderr.write(
            "[biomevae/workflow] single-study mode: skipping cross-study "
            "models: {names}\n".format(names=", ".join(sorted(_dropped)))
        )

ALL_MODELS = list(MODELS.keys())


wildcard_constraints:
    study = r"[A-Za-z0-9_\-\.]+",
    model = r"[A-Za-z0-9_\-]+",


# ── path helpers ────────────────────────────────────────────────────
def data_path(study, filename):
    return os.path.join(DATA_ROOT, study, filename)

def study_out(study):
    return os.path.join(OUTPUT_ROOT, study)

def models_dir(study):
    return os.path.join(OUTPUT_ROOT, study, "models")

def model_dir(study, model):
    return os.path.join(OUTPUT_ROOT, study, "models", model)

def figures_dir(study):
    return os.path.join(OUTPUT_ROOT, study, "figures")

def aggregate_dir(study):
    return os.path.join(OUTPUT_ROOT, study, "models", "aggregate")

def display_name(study):
    return STUDY_DISPLAY_NAMES.get(study, study)


# ── multi-label study selector (for the meta pipeline) ─────────────
def select_multi_label_studies(min_labels=2, require_case_control=True,
                               body_site=None, only=None, exclude=None):
    """Return the set of registry studies that qualify as multi-label.

    A study is *multi-label* when its ``disease_labels`` has at least
    ``min_labels`` entries, which is the minimum requirement to train a
    classifier.  By default we also require ``has_case_control=True`` so
    that the meta pipeline focuses on studies with a real disease-versus-
    control contrast.
    """
    try:
        from curatedmetagenomicdata.study_registry import STUDY_REGISTRY
    except ImportError as exc:
        raise ImportError(
            "Could not import 'curatedmetagenomicdata.study_registry'. "
            "Set 'extract_microbiome_data_path' in config to point at a "
            "checkout of extract-microbiome-data."
        ) from exc

    only_set = set(only) if only else None
    exclude_set = set(exclude) if exclude else set()

    selected = []
    for name, info in STUDY_REGISTRY.items():
        if only_set is not None and name not in only_set:
            continue
        if name in exclude_set:
            continue
        if len(info.disease_labels) < min_labels:
            continue
        if require_case_control and not info.has_case_control:
            continue
        if body_site and info.body_site != body_site:
            continue
        selected.append(name)
    return sorted(selected)


def resolve_studies():
    """Turn the Snakemake ``config`` into a list of studies to run."""
    if config.get("auto_multi_label", False):
        candidates = select_multi_label_studies(
            min_labels=int(config.get("min_labels", 2)),
            require_case_control=bool(config.get("require_case_control", True)),
            body_site=config.get("body_site"),
            only=config.get("only_studies"),
            exclude=config.get("exclude_studies"),
        )
    elif "studies" in config:
        candidates = list(config["studies"])
    elif "study" in config:
        candidates = [config["study"]]
    else:
        raise ValueError(
            "Config must set one of: 'study' (single name), 'studies' "
            "(list), or 'auto_multi_label: true' (meta pipeline)."
        )

    # Only keep studies whose extracted inputs exist on disk.  This lets
    # the meta pipeline run against a partially-downloaded registry
    # without blowing up.
    verified, missing = [], []
    for s in candidates:
        required = [
            data_path(s, "sgb_table.tsv"),
            data_path(s, "phyla.tsv"),
            data_path(s, "sample_metadata.tsv"),
        ]
        if all(os.path.exists(p) for p in required):
            verified.append(s)
        else:
            missing.append(s)

    if missing:
        sys.stderr.write(
            "[biomevae/workflow] Skipping {n} studies without extracted "
            "inputs under {root}: {names}\n".format(
                n=len(missing), root=DATA_ROOT, names=", ".join(missing)
            )
        )

    if not verified:
        raise FileNotFoundError(
            f"None of the requested studies have extracted inputs under "
            f"{DATA_ROOT}. Run extract-microbiome-data first, or point "
            f"'data_root' at the correct directory."
        )

    return verified


STUDIES = resolve_studies()
