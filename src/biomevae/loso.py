"""Leave-One-Study-Out utilities for the biomevae pipeline.

Three building blocks live here:

1. :func:`merge_studies` — concatenate per-study ``sgb_table.tsv`` /
   ``phyla.tsv`` / ``sample_metadata.tsv`` into a single multi-study
   dataset with a unified feature space (union of clades, zero-fill
   missing) and a ``study_name`` column on the metadata table.

2. :func:`leave_one_study_out_splits` — yield ``(train_mask, eval_mask,
   held_out_study)`` triples for a list of studies.  Used by both the
   Snakemake DAG and the plain-Python CLI runner.

3. :class:`ControlAnchor` — diagnostics that quantify
   distribution drift between studies *within the control class*:
   per-pair Frobenius distance between latent covariance matrices
   (CORAL diagnostic) and unbiased Maximum Mean Discrepancy with a
   multi-bandwidth Gaussian kernel.  These are the "is domain
   adaptation even needed?" numbers — see ``workflow/README_loso.md``
   for the calling convention.

The module is deliberately PyTorch-free at top level so it can be
imported in light-weight Snakemake driver rules.  The MMD helper takes
raw NumPy arrays.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


__all__ = [
    "ControlAnchor",
    "MergedStudies",
    "control_anchor",
    "coral_align",
    "covariance_frobenius",
    "leave_one_study_out_splits",
    "load_merged",
    "merge_studies",
    "mmd_rbf_unbiased",
]


# ---------------------------------------------------------------------------
# Multi-study merging
# ---------------------------------------------------------------------------


@dataclass
class MergedStudies:
    """Bundle returned by :func:`merge_studies`.

    Attributes
    ----------
    sgb_table:
        ``DataFrame`` with the same schema as a per-study
        ``sgb_table.tsv``: ``clade_name``, ``NCBI_tax_id`` then one
        column per sample.  Missing clades are zero-filled.
    phyla:
        Concatenated taxonomy table, deduplicated on ``clade_name``.
        Identical to the original schema (no header row in the source
        files; we preserve that on write).
    metadata:
        Per-sample metadata with a ``study_name`` column added /
        verified.  Indexed by ``sample_id``.
    feature_clades:
        The union of clades across studies, used as the row order of
        ``sgb_table`` and the column order downstream.
    sample_to_study:
        Mapping from ``sample_id`` to ``study_name``.  Convenience for
        :func:`leave_one_study_out_splits` consumers.
    """

    sgb_table: pd.DataFrame
    phyla: pd.DataFrame
    metadata: pd.DataFrame
    feature_clades: List[str]
    sample_to_study: Dict[str, str]

    def write(self, outdir: os.PathLike | str) -> Dict[str, str]:
        """Persist as the standard three-file layout under ``outdir``."""
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        sgb_path = out / "sgb_table.tsv"
        phyla_path = out / "phyla.tsv"
        meta_path = out / "sample_metadata.tsv"
        self.sgb_table.to_csv(sgb_path, sep="\t", index=False)
        # phyla.tsv is conventionally header-less in the rest of the
        # pipeline; preserve that.
        self.phyla.to_csv(phyla_path, sep="\t", index=False, header=False)
        meta = self.metadata.reset_index().rename(columns={"index": "sample_id"})
        if "sample_id" not in meta.columns:
            meta.insert(0, "sample_id", self.metadata.index)
        meta.to_csv(meta_path, sep="\t", index=False)
        return {
            "sgb_table": str(sgb_path),
            "phyla": str(phyla_path),
            "sample_metadata": str(meta_path),
        }


def _read_sgb_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if df.shape[1] < 3:
        raise ValueError(
            f"{path}: expected clade_name, NCBI_tax_id and >=1 sample columns."
        )
    return df


def _read_phyla(path: Path) -> pd.DataFrame:
    """phyla.tsv is convention-less; assume no header."""
    return pd.read_csv(path, sep="\t", header=None, dtype=str)


def _read_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "sample_id" not in df.columns:
        # Some upstream extractors put the ID in the index column.
        df = df.rename(columns={df.columns[0]: "sample_id"})
    return df.set_index("sample_id")


def merge_studies(
    data_root: os.PathLike | str,
    studies: Sequence[str],
    *,
    require_columns: Sequence[str] = ("disease",),
    study_col: str = "study_name",
) -> MergedStudies:
    """Concatenate per-study extracted files into a single dataset.

    Each ``<data_root>/<study>`` folder must contain ``sgb_table.tsv``,
    ``phyla.tsv`` and ``sample_metadata.tsv`` — the canonical layout
    produced by ``extract-microbiome-data``.

    The merged ``sgb_table`` uses the *union* of clades observed across
    studies, with zero-fill for clades absent from a given study.  The
    merged ``phyla`` is the concatenation of the per-study taxonomy
    tables, deduplicated on ``clade_name``.  Sample IDs are kept
    unchanged but a ``study_name`` column is force-set on the metadata
    so downstream LOSO splitting is unambiguous.
    """
    if not studies:
        raise ValueError("merge_studies: 'studies' must be non-empty.")

    root = Path(data_root)
    sample_columns_per_study: Dict[str, List[str]] = {}
    sgb_per_study: Dict[str, pd.DataFrame] = {}
    phyla_frames: List[pd.DataFrame] = []
    metadata_frames: List[pd.DataFrame] = []

    all_clades: List[str] = []
    seen_clades: set[str] = set()

    for s in studies:
        sdir = root / s
        sgb_p = sdir / "sgb_table.tsv"
        phyla_p = sdir / "phyla.tsv"
        meta_p = sdir / "sample_metadata.tsv"
        for p in (sgb_p, phyla_p, meta_p):
            if not p.exists():
                raise FileNotFoundError(
                    f"merge_studies: {p} not found for study '{s}'."
                )

        sgb = _read_sgb_table(sgb_p)
        # Preserve the union ordering — keep the first occurrence of each
        # clade across studies for stable downstream feature ordering.
        for c in sgb["clade_name"].astype(str):
            if c not in seen_clades:
                all_clades.append(c)
                seen_clades.add(c)
        sgb_per_study[s] = sgb
        sample_columns_per_study[s] = list(sgb.columns[2:])

        phyla = _read_phyla(phyla_p)
        phyla_frames.append(phyla)

        meta = _read_metadata(meta_p)
        meta = meta.copy()
        meta[study_col] = s
        for col in require_columns:
            if col not in meta.columns:
                raise ValueError(
                    f"merge_studies: study {s} sample_metadata lacks "
                    f"required column '{col}'."
                )
        metadata_frames.append(meta)

    # ----- merged metadata ------------------------------------------------
    metadata = pd.concat(metadata_frames, axis=0, sort=False)
    if metadata.index.duplicated().any():
        dups = metadata.index[metadata.index.duplicated()].tolist()
        raise ValueError(
            f"merge_studies: duplicate sample IDs across studies: {dups[:5]}"
        )

    # ----- merged phyla ---------------------------------------------------
    phyla = pd.concat(phyla_frames, axis=0, sort=False)
    # Deduplicate on the first column (clade_name) preserving the first hit.
    phyla = phyla.drop_duplicates(subset=phyla.columns[0])
    # Reorder to follow ``all_clades``.
    phyla.columns = list(phyla.columns)
    phyla = phyla.set_index(phyla.columns[0]).reindex(all_clades).reset_index()

    # ----- merged sgb table ----------------------------------------------
    # Build a dense per-sample DataFrame with the union feature space.
    out_cols: List[str] = []
    out_blocks: List[np.ndarray] = []
    for s in studies:
        sgb = sgb_per_study[s].set_index("clade_name")
        ncbi = sgb["NCBI_tax_id"]
        sgb = sgb.drop(columns=["NCBI_tax_id"])
        sgb = sgb.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        # Reindex to full clade union; zero-fill missing.
        sgb = sgb.reindex(all_clades).fillna(0.0)
        out_cols.extend(sgb.columns.tolist())
        out_blocks.append(sgb.to_numpy(dtype=np.float64))
    full = np.concatenate(out_blocks, axis=1)

    # NCBI tax ids: take first non-null per clade.
    ncbi_lookup: Dict[str, str] = {}
    for s in studies:
        sgb = sgb_per_study[s]
        for clade, tax in zip(sgb["clade_name"].astype(str), sgb["NCBI_tax_id"]):
            ncbi_lookup.setdefault(clade, str(tax))
    ncbi_col = pd.Series(
        [ncbi_lookup.get(c, "") for c in all_clades], name="NCBI_tax_id",
    )
    sgb_table = pd.DataFrame(full, index=all_clades, columns=out_cols)
    sgb_table = sgb_table.reset_index().rename(columns={"index": "clade_name"})
    sgb_table.insert(1, "NCBI_tax_id", ncbi_col.values)

    sample_to_study = {
        sid: study
        for study, cols in sample_columns_per_study.items()
        for sid in cols
    }

    return MergedStudies(
        sgb_table=sgb_table,
        phyla=phyla,
        metadata=metadata,
        feature_clades=all_clades,
        sample_to_study=sample_to_study,
    )


def load_merged(
    sgb_table_path: os.PathLike | str,
    metadata_path: os.PathLike | str,
    *,
    study_col: str = "study_name",
) -> Tuple[np.ndarray, List[str], List[str], pd.DataFrame]:
    """Load a merged dataset previously written by :meth:`MergedStudies.write`.

    Returns ``(X_raw, sample_ids, feature_clades, metadata)`` where
    ``X_raw`` is shape ``(n_samples, n_features)``.
    """
    sgb = _read_sgb_table(Path(sgb_table_path))
    feature_clades = sgb["clade_name"].astype(str).tolist()
    sample_cols = sgb.columns[2:].tolist()
    X_raw = (
        sgb[sample_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        .T
    )
    metadata = _read_metadata(Path(metadata_path))
    if study_col not in metadata.columns:
        raise ValueError(
            f"load_merged: metadata lacks study column '{study_col}'."
        )
    return X_raw, sample_cols, feature_clades, metadata


# ---------------------------------------------------------------------------
# LOSO splitter
# ---------------------------------------------------------------------------


def leave_one_study_out_splits(
    sample_ids: Sequence[str],
    sample_to_study: Dict[str, str],
    *,
    only_studies: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
    """Yield ``(train_mask, eval_mask, held_out_study)`` triples.

    ``sample_ids`` must be in the order of the rows of the data matrix
    so the boolean masks index it directly.

    ``only_studies`` optionally restricts the set of studies to leave
    out (handy when the merged dataset contains studies we don't want
    to evaluate on, e.g. healthy-only cohorts).
    """
    sample_arr = np.asarray(sample_ids, dtype=object)
    if len(sample_arr) != len(sample_to_study):
        # not necessarily fatal — sample_to_study may have entries from
        # studies not represented in sample_ids — just verify coverage.
        for s in sample_arr:
            if s not in sample_to_study:
                raise KeyError(
                    f"leave_one_study_out_splits: sample '{s}' not in "
                    f"sample_to_study lookup."
                )

    studies = sorted({sample_to_study[s] for s in sample_arr})
    if only_studies is not None:
        only_set = set(only_studies)
        studies = [s for s in studies if s in only_set]

    studies_per_sample = np.array(
        [sample_to_study[s] for s in sample_arr], dtype=object,
    )
    for held in studies:
        eval_mask = studies_per_sample == held
        train_mask = ~eval_mask
        if not eval_mask.any():
            continue
        if not train_mask.any():
            raise ValueError(
                f"leave_one_study_out_splits: leaving out '{held}' empties "
                f"the training set."
            )
        yield train_mask, eval_mask, held


# ---------------------------------------------------------------------------
# Control-anchor diagnostics: covariance Frobenius distance + MMD
# ---------------------------------------------------------------------------


def covariance_frobenius(
    X: np.ndarray, Y: np.ndarray, *, ddof: int = 1,
) -> float:
    """Frobenius distance between sample covariance matrices of two sets.

    This is the CORAL discrepancy (Sun & Saenko 2016) without the
    ``1/(4 d^2)`` normalisation so the value is on the same scale as
    ``||Σ_X - Σ_Y||_F`` reported in the literature.
    """
    if X.size == 0 or Y.size == 0:
        return float("nan")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "covariance_frobenius: X and Y must have the same column count."
        )
    cov_x = np.cov(X, rowvar=False, ddof=ddof)
    cov_y = np.cov(Y, rowvar=False, ddof=ddof)
    return float(np.linalg.norm(cov_x - cov_y, ord="fro"))


def _psd_sqrt(
    cov: np.ndarray, *, ridge: float, invert: bool,
) -> np.ndarray:
    """Symmetric (positive) square root or inverse-square-root of a covariance.

    ``ridge`` is added to the diagonal as a fraction of ``tr(cov)/p`` before
    decomposition, so the matrix is well-conditioned even when the study has
    fewer samples than features (Hannigan / Gupta cohorts have <100 samples
    and >1000 features in the merged CRC table).  Eigenvalues are clipped to
    a small floor before exponentiation to keep the inverse stable.
    """
    p = cov.shape[0]
    trace_avg = float(np.trace(cov)) / max(p, 1)
    reg = cov + ridge * trace_avg * np.eye(p, dtype=cov.dtype)
    # Symmetrise to absorb any floating-point asymmetry from np.cov on
    # sparse-microbiome inputs.
    reg = 0.5 * (reg + reg.T)
    eigvals, eigvecs = np.linalg.eigh(reg)
    floor = max(np.finfo(cov.dtype).eps * float(eigvals.max(initial=1.0)), 1e-12)
    eigvals = np.clip(eigvals, floor, None)
    if invert:
        s = 1.0 / np.sqrt(eigvals)
    else:
        s = np.sqrt(eigvals)
    return (eigvecs * s) @ eigvecs.T


def coral_align(
    X: np.ndarray,
    study: Sequence[str],
    *,
    ridge: float = 1e-3,
    reference: str = "mean",
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """Per-study CORAL alignment of a feature matrix.

    For each study ``s`` we compute a regularised mean μ_s and covariance
    Σ_s, then re-colour every sample to a shared reference distribution
    (μ_ref, Σ_ref)::

        x' = (x - μ_s) · Σ_s^{-1/2} · Σ_ref^{1/2} + μ_ref

    The transform for a sample depends only on its own study's statistics, so
    the held-out study in a leave-one-study-out split is aligned with its own
    (unsupervised) features and no class label leaks across the boundary.
    This is the unsupervised CORAL of Sun & Saenko (2016) extended to many
    sources by aligning every study to a common reference.

    Parameters
    ----------
    X
        ``(n_samples, n_features)`` matrix.  Caller is responsible for any
        prior transform (e.g. ``log1p``).
    study
        Per-sample study label, length ``n_samples``.
    ridge
        Diagonal regularisation as a fraction of ``tr(Σ_s)/p`` added to each
        per-study covariance before eigendecomposition.  Required when
        ``n_samples_s < n_features``.  Default ``1e-3``.
    reference
        How to pick the shared target distribution:

        * ``"mean"`` (default) — element-wise mean of per-study μ and Σ
          (after ridge regularisation).  Symmetric across studies.
        * ``"identity"`` — μ_ref = 0, Σ_ref = I.  Equivalent to whitening
          every study to a unit Gaussian.
        * ``"largest"`` — pick the study with the most samples as reference.
          Closest to "align everything to the dominant cohort".

    Returns
    -------
    X_aligned
        Same shape as ``X``, dtype-preserved (or upcast to float64 if input
        was integer).
    stats
        ``{study_name: {"mu": np.ndarray, "cov": np.ndarray, "n": int}}``
        for diagnostics.  Also contains ``"_reference": {"mu": ..., "cov":
        ...}`` for the chosen target.
    """
    if X.ndim != 2:
        raise ValueError(f"coral_align: X must be 2D; got shape {X.shape}.")
    study_arr = np.asarray(study, dtype=object)
    if study_arr.shape[0] != X.shape[0]:
        raise ValueError(
            f"coral_align: len(study)={study_arr.shape[0]} does not match "
            f"X.shape[0]={X.shape[0]}."
        )
    if reference not in ("mean", "identity", "largest"):
        raise ValueError(
            "coral_align: reference must be one of "
            "{'mean', 'identity', 'largest'}; got " + repr(reference) + "."
        )

    studies = sorted({str(s) for s in study_arr})
    p = X.shape[1]
    dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64

    per_study: Dict[str, dict] = {}
    for s in studies:
        mask = study_arr == s
        n_s = int(mask.sum())
        Xs = X[mask].astype(dtype, copy=False)
        if n_s < 2:
            # A single sample carries no covariance information; default to
            # identity covariance so the alignment becomes a pure shift.
            mu_s = Xs.mean(axis=0) if n_s == 1 else np.zeros(p, dtype=dtype)
            cov_s = np.eye(p, dtype=dtype)
        else:
            mu_s = Xs.mean(axis=0)
            cov_s = np.cov(Xs, rowvar=False, ddof=1).astype(dtype, copy=False)
            if cov_s.ndim == 0:
                cov_s = cov_s.reshape(1, 1)
        per_study[s] = {"mu": mu_s, "cov": cov_s, "n": n_s}

    if reference == "identity":
        mu_ref = np.zeros(p, dtype=dtype)
        cov_ref = np.eye(p, dtype=dtype)
    elif reference == "largest":
        ref_name = max(studies, key=lambda s: per_study[s]["n"])
        mu_ref = per_study[ref_name]["mu"].copy()
        cov_ref = per_study[ref_name]["cov"].copy()
    else:  # "mean"
        mu_ref = np.mean(
            np.stack([per_study[s]["mu"] for s in studies], axis=0), axis=0,
        )
        cov_ref = np.mean(
            np.stack([per_study[s]["cov"] for s in studies], axis=0), axis=0,
        )

    cov_ref_sqrt = _psd_sqrt(cov_ref, ridge=ridge, invert=False)

    X_aligned = np.empty(X.shape, dtype=dtype)
    for s in studies:
        mask = study_arr == s
        if not mask.any():
            continue
        Xs = X[mask].astype(dtype, copy=False)
        whitener = _psd_sqrt(per_study[s]["cov"], ridge=ridge, invert=True)
        X_aligned[mask] = (Xs - per_study[s]["mu"]) @ whitener @ cov_ref_sqrt + mu_ref

    per_study["_reference"] = {"mu": mu_ref, "cov": cov_ref}
    return X_aligned, per_study


def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = (A * A).sum(axis=1, keepdims=True)
    b2 = (B * B).sum(axis=1, keepdims=True).T
    return a2 + b2 - 2.0 * (A @ B.T)


def mmd_rbf_unbiased(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    bandwidths: Optional[Sequence[float]] = None,
) -> float:
    """Unbiased MMD² with a sum-of-Gaussians kernel (Gretton et al. 2012).

    The default bandwidths are the median heuristic on the pooled data
    multiplied by ``[0.25, 0.5, 1, 2, 4]``, which matches the standard
    MMD-VAE / DIVA practice.

    Returns the unbiased squared-MMD estimator.  Negative values are
    possible for small samples and indicate the two distributions are
    statistically indistinguishable in the kernel feature space; we do
    not clip them so callers can see the noise floor.
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("mmd_rbf_unbiased: X and Y must have the same dim.")
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return float("nan")

    XX = _pairwise_sq_dists(X, X)
    YY = _pairwise_sq_dists(Y, Y)
    XY = _pairwise_sq_dists(X, Y)

    if bandwidths is None:
        all_d = np.concatenate([
            XX[np.triu_indices(n, k=1)],
            YY[np.triu_indices(m, k=1)],
            XY.ravel(),
        ])
        med = float(np.median(all_d))
        if med <= 0:
            med = 1.0
        bandwidths = [med * f for f in (0.25, 0.5, 1.0, 2.0, 4.0)]

    total = 0.0
    for sigma_sq in bandwidths:
        gamma = 1.0 / (2.0 * sigma_sq)
        k_xx = np.exp(-gamma * XX)
        k_yy = np.exp(-gamma * YY)
        k_xy = np.exp(-gamma * XY)
        # Unbiased: drop the diagonal of the within-set kernels.
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)
        mmd2 = (
            k_xx.sum() / (n * (n - 1))
            + k_yy.sum() / (m * (m - 1))
            - 2.0 * k_xy.mean()
        )
        total += float(mmd2)
    return total / float(len(bandwidths))


@dataclass
class ControlAnchor:
    """Cross-study drift diagnostics restricted to a reference (control) class.

    The two metrics quantify how far apart the *control* distributions
    of every pair of studies are after embedding into ``z``-space:

    * :attr:`coral_pairs`   — Frobenius covariance distance, the CORAL
      discrepancy.  Cheap, second-moment, and a sanity check that
      DIVA / MMD regularisers are doing their job.
    * :attr:`mmd_pairs`     — multi-bandwidth Gaussian-kernel MMD²,
      higher-order, and the canonical "are these two distributions
      different" test.

    Pairs are stored as a long-format DataFrame so they can be melted
    into a heatmap or merged with downstream LOSO scores.
    """

    coral_pairs: pd.DataFrame   # cols: study_a, study_b, frobenius
    mmd_pairs: pd.DataFrame     # cols: study_a, study_b, mmd2
    studies: List[str]

    def to_tsv(self, outdir: os.PathLike | str) -> Dict[str, str]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        coral_p = out / "control_anchor_coral.tsv"
        mmd_p = out / "control_anchor_mmd.tsv"
        self.coral_pairs.to_csv(coral_p, sep="\t", index=False)
        self.mmd_pairs.to_csv(mmd_p, sep="\t", index=False)
        return {"coral": str(coral_p), "mmd": str(mmd_p)}


def control_anchor(
    Z: np.ndarray,
    sample_ids: Sequence[str],
    metadata: pd.DataFrame,
    *,
    label_col: str = "disease",
    control_value: str = "healthy",
    study_col: str = "study_name",
    min_samples: int = 5,
) -> ControlAnchor:
    """Compute pair-wise control-only CORAL and MMD between every pair of studies.

    Parameters
    ----------
    Z:
        ``(n_samples, latent_dim)`` embedding matrix; rows aligned to
        ``sample_ids``.
    sample_ids:
        Row labels of ``Z``.
    metadata:
        DataFrame indexed by ``sample_id`` with at least ``label_col``
        and ``study_col``.
    control_value:
        Value of ``label_col`` that defines the reference class.  Most
        cMD studies use ``"healthy"`` or ``"control"``; both are accepted
        if you pass a tuple — use a custom mapping upstream if the
        registry uses something else.
    min_samples:
        Studies with fewer than this many control samples are excluded
        — covariance estimates are noise-dominated below ~5.
    """
    Z = np.asarray(Z, dtype=np.float64)
    sample_arr = list(sample_ids)
    by_sample = metadata.reindex(sample_arr)

    is_control = by_sample[label_col].astype(str).str.lower() == control_value.lower()
    studies = by_sample[study_col].astype(str).values

    per_study: Dict[str, np.ndarray] = {}
    for s in sorted(set(studies)):
        rows = (studies == s) & is_control.values
        if rows.sum() >= min_samples:
            per_study[s] = Z[rows]

    keep = sorted(per_study.keys())
    coral_rows = []
    mmd_rows = []
    for i, sa in enumerate(keep):
        for sb in keep[i + 1 :]:
            coral_rows.append({
                "study_a": sa, "study_b": sb,
                "frobenius": covariance_frobenius(per_study[sa], per_study[sb]),
                "n_a": int(per_study[sa].shape[0]),
                "n_b": int(per_study[sb].shape[0]),
            })
            mmd_rows.append({
                "study_a": sa, "study_b": sb,
                "mmd2": mmd_rbf_unbiased(per_study[sa], per_study[sb]),
                "n_a": int(per_study[sa].shape[0]),
                "n_b": int(per_study[sb].shape[0]),
            })

    return ControlAnchor(
        coral_pairs=pd.DataFrame(coral_rows),
        mmd_pairs=pd.DataFrame(mmd_rows),
        studies=keep,
    )
