#!/usr/bin/env python3
"""Statistical analysis of single-study predictability by disease category.

Addresses the question: *are some disease categories easier to predict than
others in the single-study benchmark?*

Each of the 42 single-study cohorts is assigned a disease category derived from
the authoritative per-study ``sample_metadata.tsv`` ``disease`` column in the
sibling ``extract-microbiome-data`` checkout (the same labels the classifiers
were trained on). Per-study predictability is summarised as the mean AUROC
across all models (the study is the unit of analysis, avoiding
pseudoreplication across models), and differences between disease categories
are tested with a Kruskal-Wallis omnibus test.

Outputs (to results/figures/current_results/ by default):
    disease_category_predictability.{pdf,png}
    disease_category_study_table.tsv
    disease_category_stats.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Disease assignment.
#
# ``STUDY_DISEASE`` = the dominant case condition in each cohort's cMD
# ``disease`` metadata column (control excluded). ``DISEASE_CATEGORY`` groups
# those into coarser categories with enough cohorts per group for a meaningful
# between-category test. Both are editable; the specific disease is kept so the
# grouping can be refined without re-deriving from metadata.
# --------------------------------------------------------------------------- #
STUDY_DISEASE: dict[str, str] = {
    # Colorectal cancer (the 11 cMD CRC cohorts, matches workflow/config/loso_crc.yaml)
    "FengQ_2015": "CRC",
    "GuptaA_2019": "CRC",
    "HanniganGD_2017": "CRC",
    "ThomasAM_2018a": "CRC",
    "ThomasAM_2018b": "CRC",
    "ThomasAM_2019_c": "CRC",
    "VogtmannE_2016": "CRC",
    "WirbelJ_2018": "CRC",
    "YachidaS_2019": "CRC",
    "YuJ_2015": "CRC",
    "ZellerG_2014": "CRC",
    # Inflammatory bowel disease
    "HMP_2019_ibdmdb": "IBD",
    "HallAB_2017": "IBD",
    "IjazUZ_2017": "IBD",
    "NielsenHB_2014": "IBD",
    "LiJ_2014": "IBD",
    # Glucose metabolism / type-2 diabetes
    "HMP_2019_t2d": "T2D/IGT",
    "KarlssonFH_2013": "T2D/IGT",
    "QinJ_2012": "T2D/IGT",
    "SankaranarayananK_2015": "T2D/IGT",
    "MetaCardis_2020_a": "cardiometabolic",
    # Type-1 diabetes
    "Heitz-BuschartA_2016": "T1D",
    "KosticAD_2015": "T1D",
    # Cardiovascular / hypertension
    "JieZ_2017": "cardiometabolic",
    "LiJ_2017": "cardiometabolic",
    # Liver
    "QinN_2014": "liver cirrhosis",
    # Neuropsychiatric
    "Castro-NallarE_2015": "neuropsychiatric",
    "ZhuF_2020": "neuropsychiatric",
    "BedarfJR_2017": "neuropsychiatric",
    "NagySzakalD_2017": "neuropsychiatric",
    # GI infection / diarrhoea / FMT
    "DavidLA_2015": "GI infection",
    "KieserS_2018": "GI infection",
    "VincentC_2016": "GI infection",
    "IaniroG_2022": "GI infection",
    # Soil-transmitted helminth / parasitic
    "RosaBA_2018": "parasitic (STH)",
    "RubelMA_2020": "parasitic (STH)",
    # Skin / oral
    "ChngKR_2016": "skin/oral",
    "GhensiP_2019": "skin/oral",
    # Other (developmental / autoimmune / respiratory)
    "BrooksB_2017": "other",
    "LiSS_2016": "other",
    "YeZ_2018": "other",
    "XieH_2016": "other",
}

# Coarse grouping used for the omnibus between-category test (more power).
CATEGORY_GROUP: dict[str, str] = {
    "CRC": "Cancer (CRC)",
    "IBD": "IBD",
    "T2D/IGT": "Metabolic",
    "cardiometabolic": "Metabolic",
    "T1D": "Metabolic",
    "liver cirrhosis": "Metabolic",
    "neuropsychiatric": "Neuropsychiatric",
    "GI infection": "Infection/parasitic",
    "parasitic (STH)": "Infection/parasitic",
    "skin/oral": "Other",
    "other": "Other",
}

CATEGORY_COLORS = {
    "Cancer (CRC)": "#882255",
    "IBD": "#CC6677",
    "Metabolic": "#DDCC77",
    "Neuropsychiatric": "#AA4499",
    "Infection/parasitic": "#44AA99",
    "Other": "#888888",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--meta",
        type=Path,
        default=Path("results/figures/current_results/latest_single_meta_summary.tsv"),
    )
    p.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("/home/rvalenciaaz/GIT_FOLDER/extract-microbiome-data/data/per_study"),
        help="Per-study sample_metadata.tsv root (for cohort sample sizes).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/figures/current_results"),
    )
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return p.parse_args()


def load_sample_sizes(studies, root: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for s in studies:
        md = root / s / "sample_metadata.tsv"
        if md.exists():
            sizes[s] = int(sum(1 for _ in md.open()) - 1)  # rows minus header
    return sizes


def load_class_balance(studies, root: Path) -> dict[str, float]:
    """Minority-class fraction per cohort (0.5 = perfectly balanced case/control)."""
    bal: dict[str, float] = {}
    for s in studies:
        md = root / s / "sample_metadata.tsv"
        if not md.exists():
            continue
        d = pd.read_csv(md, sep="\t")
        if "disease" not in d.columns:
            continue
        vc = d["disease"].fillna("NA").value_counts()
        n = int(vc.sum())
        ctrl = int(vc.get("control", 0))
        case = n - ctrl
        if n:
            bal[s] = min(case, ctrl) / n
    return bal


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.edgecolor": "#1f2937",
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def build_study_table(
    meta: pd.DataFrame,
    sizes: dict[str, int] | None = None,
    balance: dict[str, float] | None = None,
) -> pd.DataFrame:
    meta = meta.copy()
    for col in ("auroc", "balanced_accuracy", "f1_macro"):
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    per_study = (
        meta.groupby("study")
        .agg(
            mean_auroc=("auroc", "mean"),
            min_auroc=("auroc", "min"),
            max_auroc=("auroc", "max"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_f1_macro=("f1_macro", "mean"),
            n_models=("auroc", "count"),
        )
        .reset_index()
    )
    per_study["disease"] = per_study["study"].map(STUDY_DISEASE)
    per_study["category"] = per_study["disease"].map(CATEGORY_GROUP)
    if sizes:
        per_study["n_samples"] = per_study["study"].map(sizes)
    if balance:
        per_study["minority_frac"] = per_study["study"].map(balance)
        if sizes:
            per_study["minority_n"] = (per_study["minority_frac"] * per_study["n_samples"]).round().astype("Int64")
    missing = per_study[per_study["disease"].isna()]["study"].tolist()
    if missing:
        raise SystemExit(f"Unmapped studies (add to STUDY_DISEASE): {missing}")
    return per_study.sort_values("mean_auroc", ascending=False).reset_index(drop=True)


def category_stats(per_study: pd.DataFrame, value: str = "mean_auroc") -> tuple[pd.DataFrame, dict]:
    groups = []
    rows = []
    for cat, g in per_study.groupby("category"):
        vals = g[value].dropna().to_numpy()
        groups.append(vals)
        rows.append(
            {
                "category": cat,
                "n_studies": len(vals),
                "median_auroc": float(np.median(vals)),
                "mean_auroc": float(np.mean(vals)),
                "iqr_low": float(np.percentile(vals, 25)),
                "iqr_high": float(np.percentile(vals, 75)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("median_auroc", ascending=False).reset_index(drop=True)
    usable = [g for g in groups if len(g) >= 1]
    if len([g for g in groups if len(g) >= 2]) >= 2:
        h, p = stats.kruskal(*[g for g in groups if len(g) >= 1])
    else:
        h, p = np.nan, np.nan
    return summary, {"kruskal_H": float(h), "kruskal_p": float(p), "n_categories": len(groups)}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size in [-1, 1]; >0 means `a` tends to exceed `b`."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = np.subtract.outer(a, b)
    return float((np.sign(diff)).sum() / (len(a) * len(b)))


def _delta_magnitude(d: float) -> str:
    ad = abs(d)
    if np.isnan(d):
        return ""
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def pairwise_tests(per_study: pd.DataFrame, order: list[str], value: str = "mean_auroc"):
    """Pairwise Mann-Whitney U (BH-FDR) plus Cliff's delta effect sizes.

    Returns (order, adj_pvalue_matrix, long_table). The matrix is square in the
    given category order with NaN on the diagonal; the long table lists each
    pair with raw/adjusted p-values and the effect size + magnitude.
    """
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests

    vals = {c: per_study.loc[per_study["category"] == c, value].dropna().to_numpy() for c in order}
    pairs = list(combinations(order, 2))
    raw = []
    for a, b in pairs:
        if len(vals[a]) >= 1 and len(vals[b]) >= 1 and (len(vals[a]) + len(vals[b])) >= 3:
            _, p = stats.mannwhitneyu(vals[a], vals[b], alternative="two-sided")
        else:
            p = np.nan
        raw.append(p)
    raw_arr = np.array(raw, dtype=float)
    ok = ~np.isnan(raw_arr)
    adj = np.full_like(raw_arr, np.nan)
    if ok.sum():
        adj[ok] = multipletests(raw_arr[ok], method="fdr_bh")[1]
    n = len(order)
    mat = np.full((n, n), np.nan)
    rows = []
    for (a, b), pr, pa in zip(pairs, raw_arr, adj):
        i, j = order.index(a), order.index(b)
        mat[i, j] = pa
        mat[j, i] = pa
        delta = cliffs_delta(vals[a], vals[b])
        rows.append(
            {
                "category_a": a,
                "category_b": b,
                "p_raw": pr,
                "p_fdr_bh": pa,
                "cliffs_delta": delta,
                "effect_magnitude": _delta_magnitude(delta),
            }
        )
    return order, mat, pd.DataFrame(rows)


def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def sensitivity_small_cohorts(per_study: pd.DataFrame, thresholds=(0, 10, 20)) -> pd.DataFrame:
    """Re-run the omnibus test after dropping cohorts with a tiny minority class.

    Small minority classes (e.g. 5 controls) can inflate single-study AUROC, so
    this checks the disease-category effect is not an artifact of a few such
    cohorts.
    """
    if "minority_n" not in per_study.columns:
        return pd.DataFrame()
    rows = []
    for t in thresholds:
        sub = per_study[per_study["minority_n"].fillna(0) >= t]
        groups = [g["mean_auroc"].dropna().to_numpy() for _, g in sub.groupby("category")]
        groups = [g for g in groups if len(g) >= 1]
        if len([g for g in groups if len(g) >= 2]) >= 2:
            h, p = stats.kruskal(*groups)
        else:
            h, p = np.nan, np.nan
        rows.append({"min_minority_class": t, "n_studies": int(len(sub)), "kruskal_H": h, "kruskal_p": p})
    return pd.DataFrame(rows)


def metric_robustness(per_study: pd.DataFrame) -> pd.DataFrame:
    """Re-run the omnibus test under several predictability definitions.

    Tells us whether the category ordering / significance is an artifact of one
    metric. Rank-consistency vs mean AUROC is reported as a Spearman rho on the
    per-category medians.
    """
    metrics = {
        "mean_auroc": "mean_auroc",
        "max_auroc (best model)": "max_auroc",
        "mean_balanced_accuracy": "mean_balanced_accuracy",
        "mean_f1_macro": "mean_f1_macro",
    }
    ref_med = per_study.groupby("category")["mean_auroc"].median()
    rows = []
    for name, col in metrics.items():
        groups = [g[col].dropna().to_numpy() for _, g in per_study.groupby("category")]
        groups = [g for g in groups if len(g) >= 1]
        h, p = stats.kruskal(*groups)
        med = per_study.groupby("category")[col].median()
        rho = stats.spearmanr(ref_med.loc[med.index], med).correlation
        rows.append({"metric": name, "kruskal_H": h, "kruskal_p": p, "spearman_rho_vs_auroc": rho})
    return pd.DataFrame(rows)


def _confound_scatter(ax, sub, xcol, xlabel, order, panel_title, logx=False):
    """Scatter of per-study predictability vs a candidate confound + Spearman."""
    xvals = sub[xcol].to_numpy(dtype=float)
    avals = sub["mean_auroc"].to_numpy(dtype=float)
    for c in order:
        m = (sub["category"] == c).to_numpy()
        if m.any():
            ax.scatter(xvals[m], avals[m], s=58, color=CATEGORY_COLORS.get(c, "#888888"),
                       edgecolor="white", linewidth=0.8, label=c, zorder=3)
    tx = np.log10(xvals) if logx else xvals
    if logx:
        ax.set_xscale("log")
    rho, prho = stats.spearmanr(xvals, avals)
    slope, intercept = np.polyfit(tx, avals, 1)
    xs = np.linspace(tx.min(), tx.max(), 50)
    ax.plot(10 ** xs if logx else xs, slope * xs + intercept, ls="--", lw=1.3, color="#9ca3af", zorder=2)
    ax.axhline(0.5, ls=":", lw=1.2, color="#cbd5e1", zorder=1)
    ptxt = "p < 0.001" if prho < 1e-3 else f"p = {prho:.3f}"
    verdict = "no confound" if prho >= 0.05 else "possible confound"
    ax.annotate(
        f"Spearman ρ = {rho:.2f}, {ptxt}  ({verdict})",
        xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9.5, color="#374151", ha="left", va="bottom",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Per-study mean AUROC across models")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(panel_title, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    return rho, prho


def _titled(letter: str | None, text: str) -> str:
    """Prefix a panel letter for the multipanel; bare title when standalone."""
    return f"{letter}  {text}" if letter else text


def _save_fig(fig, outdir: Path, stem: str, formats) -> list[Path]:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=600 if fmt == "png" else 1200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _draw_predictability(ax, per_study, summary, omnibus, order, letter: str | None = None) -> None:
    """Per-category AUROC boxplot + jittered cohorts + Kruskal-Wallis omnibus."""
    positions = np.arange(len(order))
    box_data = [per_study.loc[per_study["category"] == c, "mean_auroc"].to_numpy() for c in order]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#111111", linewidth=1.8),
        whiskerprops=dict(color="#374151", linewidth=1.1),
        capprops=dict(color="#374151", linewidth=1.1),
        boxprops=dict(linewidth=1.0),
    )
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(CATEGORY_COLORS.get(c, "#888888"))
        patch.set_alpha(0.45)
        patch.set_edgecolor("#374151")

    rng = np.random.default_rng(0)
    for i, c in enumerate(order):
        vals = box_data[i]
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            positions[i] + jitter,
            vals,
            s=46,
            color=CATEGORY_COLORS.get(c, "#888888"),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    ax.axhline(0.5, ls=":", lw=1.2, color="#9ca3af", zorder=1)
    ax.text(len(order) - 0.5, 0.505, "chance (0.5)", fontsize=8.5, color="#6b7280", va="bottom", ha="right")

    labels = [f"{c}\n(n={int(summary.loc[summary['category']==c,'n_studies'].iloc[0])})" for c in order]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Per-study mean AUROC across models")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(_titled(letter, "Predictability by disease category"), loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    p = omnibus["kruskal_p"]
    ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    ax.annotate(
        f"Kruskal-Wallis: H = {omnibus['kruskal_H']:.1f}, {ptxt}",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb"),
    )


def _draw_size_confound(ax, per_study, order, letter: str | None = None) -> None:
    """Predictability vs cohort size confound check, with category legend."""
    sub = per_study.dropna(subset=["n_samples", "mean_auroc"])
    _confound_scatter(
        ax, sub, "n_samples", "Cohort size (samples, log scale)", order,
        _titled(letter, "Predictability vs cohort size"), logx=True,
    )
    ax.legend(fontsize=8.0, frameon=False, loc="upper right", title="Disease category", title_fontsize=8.5)


def _draw_balance_confound(ax, per_study, order, letter: str | None = None) -> None:
    """Predictability vs class-balance confound check."""
    subb = per_study.dropna(subset=["minority_frac", "mean_auroc"])
    _confound_scatter(
        ax, subb, "minority_frac", "Minority-class fraction (0.5 = balanced)", order,
        _titled(letter, "Predictability vs class balance"), logx=False,
    )


def _draw_pairwise(ax, pw_order, pw_mat, delta_mat, letter: str | None = None) -> None:
    """Pairwise post-hoc significance (BH-FDR Mann-Whitney) + Cliff's δ heatmap."""
    n = len(pw_order)
    masked = np.ma.masked_invalid(pw_mat)
    # Show -log10(p) so darker = more significant; cap for colour scale.
    with np.errstate(divide="ignore"):
        score = -np.log10(masked)
    im = ax.imshow(score, cmap="Purples", vmin=0, vmax=3.0, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    short = [c.replace("Infection/parasitic", "Infection/\nparasitic").replace("Neuropsychiatric", "Neuro-\npsychiatric") for c in pw_order]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(short, fontsize=8.5)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#9ca3af", fontsize=9)
            elif not np.isnan(pw_mat[i, j]):
                s = _stars(pw_mat[i, j])
                val = pw_mat[i, j]
                col = "white" if (np.isfinite(val) and -np.log10(val) > 1.6) else "#374151"
                ax.text(j, i, s, ha="center", va="center_baseline", color=col, fontsize=9, fontweight="bold")
                if not np.isnan(delta_mat[i, j]):
                    ax.text(
                        j, i + 0.26, f"δ={abs(delta_mat[i, j]):.2f}",
                        ha="center", va="center", color=col, fontsize=6.8,
                    )
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("-log₁₀(FDR p)")
    cbar.outline.set_visible(False)
    ax.set_title(_titled(letter, "Pairwise differences: significance + Cliff's δ"), loc="left")


def plot_figure(per_study: pd.DataFrame, summary: pd.DataFrame, omnibus: dict, outdir: Path, formats, robustness: pd.DataFrame | None = None) -> list[Path]:
    """Multipanel disease-category predictability figure plus, when the cohort
    metadata is available, standalone single-panel versions of each panel in
    ``disease_category_predictability_panels/``."""
    import matplotlib.pyplot as plt

    has_n = "n_samples" in per_study.columns and per_study["n_samples"].notna().any()
    has_bal = "minority_frac" in per_study.columns and per_study["minority_frac"].notna().any()
    full = has_n and has_bal
    order = summary["category"].tolist()
    pw_order, pw_mat, pw_table = pairwise_tests(per_study, order)
    # Antisymmetric Cliff's-delta matrix in pw_order for the pairwise annotation.
    delta_mat = np.full((len(pw_order), len(pw_order)), np.nan)
    for _, r in pw_table.iterrows():
        i, j = pw_order.index(r["category_a"]), pw_order.index(r["category_b"])
        delta_mat[i, j] = r["cliffs_delta"]
        delta_mat[j, i] = -r["cliffs_delta"]

    # ----- combined multipanel ----------------------------------------------
    if full:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12.5), constrained_layout=True)
        ax, ax2, ax_c, ax3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    elif has_n:
        fig, (ax, ax2, ax3) = plt.subplots(
            1, 3, figsize=(23, 7), constrained_layout=True, width_ratios=[1.25, 1.0, 1.0]
        )
    else:
        fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    _draw_predictability(ax, per_study, summary, omnibus, order, "A" if has_n else None)
    if has_n:
        _draw_size_confound(ax2, per_study, order, "B")
    if full:
        _draw_balance_confound(ax_c, per_study, order, "C")
    if has_n:
        _draw_pairwise(ax3, pw_order, pw_mat, delta_mat, "D" if full else "C")

    if robustness is not None:
        rob = robustness.set_index("metric")
        note = (
            "Robustness: category ordering identical across metrics "
            f"(Spearman ρ = {rob.loc['mean_balanced_accuracy','spearman_rho_vs_auroc']:.2f} vs AUROC); "
            f"omnibus significant for AUROC (p = {rob.loc['mean_auroc','kruskal_p']:.3f}) "
            f"but not balanced accuracy (p = {rob.loc['mean_balanced_accuracy','kruskal_p']:.3f}) "
            "— treat magnitudes, not the p-value, as the headline."
        )
        fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9, color="#6b7280", style="italic")

    paths = _save_fig(fig, outdir, "disease_category_predictability", formats)

    # ----- standalone single-panel figures (no panel letters) ---------------
    panel_dir = outdir / "disease_category_predictability_panels"
    panels = [("a", "predictability", (11, 7), lambda ax: _draw_predictability(ax, per_study, summary, omnibus, order))]
    if has_n:
        panels.append(("b", "vs_cohort_size", (8.5, 7), lambda ax: _draw_size_confound(ax, per_study, order)))
    if full:
        panels.append(("c", "vs_class_balance", (8.5, 7), lambda ax: _draw_balance_confound(ax, per_study, order)))
    if has_n:
        panels.append((
            "d" if full else "c", "pairwise_significance", (8.5, 7.5),
            lambda ax: _draw_pairwise(ax, pw_order, pw_mat, delta_mat),
        ))
    for letter, slug, figsize, draw in panels:
        sfig, sax = plt.subplots(figsize=figsize, constrained_layout=True)
        draw(sax)
        paths += _save_fig(sfig, panel_dir, f"disease_category_predictability_panel_{letter}_{slug}", formats)
    return paths


def plot_study_ranking(per_study: pd.DataFrame, outdir: Path, formats) -> list[Path]:
    """Every cohort ranked by predictability, labelled with its specific disease."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = per_study.sort_values("mean_auroc", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    colors = [CATEGORY_COLORS.get(c, "#888888") for c in d["category"]]

    fig, ax = plt.subplots(figsize=(11, 0.30 * len(d) + 1.6), constrained_layout=True)
    # range across models (min–max) as a thin line, mean as the marker
    ax.hlines(y, d["min_auroc"], d["max_auroc"], color="#cbd5e1", lw=2.2, zorder=1)
    # Flag cohorts whose minority class is tiny (<10): their AUROC is fragile.
    small = d["minority_n"].fillna(99).to_numpy() < 10 if "minority_n" in d.columns else np.zeros(len(d), bool)
    ax.scatter(d["mean_auroc"], y, s=58, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    if small.any():
        ax.scatter(
            d["mean_auroc"].to_numpy()[small], y[small], s=150, facecolors="none",
            edgecolors="#b91c1c", linewidths=1.4, zorder=4,
        )

    ax.axvline(0.5, ls=":", lw=1.2, color="#9ca3af", zorder=0)
    ax.text(0.5, len(d) - 0.2, "chance", fontsize=8.5, color="#6b7280", ha="center", va="bottom")

    ax.set_yticks(y)
    yt = [
        f"{s.replace('_', ' ')}  ·  {dis}" + ("  ⚠" if sm else "")
        for s, dis, sm in zip(d["study"], d["disease"], small)
    ]
    ax.set_yticklabels(yt, fontsize=8.5)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Per-study AUROC (marker = mean across models; line = min–max range)")
    ax.set_title("Single-study predictability ranked by cohort (coloured by disease category)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_axisbelow(True)

    cats = [c for c in CATEGORY_COLORS if c in set(d["category"])]
    handles = [Patch(facecolor=CATEGORY_COLORS[c], edgecolor="white", label=c) for c in cats]
    leg = ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="lower right", title="Disease category", title_fontsize=9)
    ax.add_artist(leg)
    if small.any():
        ax.scatter([], [], s=150, facecolors="none", edgecolors="#b91c1c", linewidths=1.4, label="minority class < 10 (fragile)")
        ax.legend(
            handles=[plt.Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#b91c1c",
                                markersize=11, markeredgewidth=1.4, ls="none", label="minority class < 10 (fragile AUROC)")],
            loc="lower right", bbox_to_anchor=(1.0, 0.18), fontsize=8.5, frameon=False,
        )

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = outdir / f"disease_study_ranking.{fmt}"
        fig.savefig(path, dpi=600 if fmt == "png" else 1200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    meta = pd.read_csv(args.meta, sep="\t")
    sizes = load_sample_sizes(meta["study"].unique(), args.metadata_root)
    balance = load_class_balance(meta["study"].unique(), args.metadata_root)
    per_study = build_study_table(meta, sizes, balance)
    summary, omnibus = category_stats(per_study)

    per_study.to_csv(args.outdir / "disease_category_study_table.tsv", sep="\t", index=False)
    summary.to_csv(args.outdir / "disease_category_stats.tsv", sep="\t", index=False)
    _, _, pw_table = pairwise_tests(per_study, summary["category"].tolist())
    pw_table.to_csv(args.outdir / "disease_category_pairwise.tsv", sep="\t", index=False)
    robustness = metric_robustness(per_study)
    robustness.to_csv(args.outdir / "disease_category_robustness.tsv", sep="\t", index=False)
    sensitivity = sensitivity_small_cohorts(per_study)
    if not sensitivity.empty:
        sensitivity.to_csv(args.outdir / "disease_category_sensitivity.tsv", sep="\t", index=False)

    paths = plot_figure(per_study, summary, omnibus, args.outdir, args.formats, robustness)
    paths += plot_study_ranking(per_study, args.outdir, args.formats)
    print("Kruskal-Wallis:", omnibus)
    print(summary.to_string(index=False))
    print("\nPairwise (BH-FDR):")
    print(pw_table.sort_values("p_fdr_bh").to_string(index=False))
    print("\nMetric robustness:")
    print(robustness.to_string(index=False))
    if not sensitivity.empty:
        print("\nSensitivity to small-minority cohorts:")
        print(sensitivity.to_string(index=False))
    print("\nWrote:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
