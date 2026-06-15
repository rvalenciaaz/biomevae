#!/usr/bin/env python3
"""Statistical tests for the strict leave-one-study-out (LOSO) CRC benchmark.

Strict LOSO holds out one colorectal-cancer cohort at a time and trains on the
other ten, measuring cross-cohort generalisation. The descriptive ranking
figure shows *which* model scores highest but tests none of the questions the
design exists to answer. This script adds those tests.

Panel implemented so far:
  A. Does domain adaptation help? Paired Wilcoxon signed-rank of each
     domain-adaptation variant against its plain backbone, across the 11
     held-out cohorts (the blocks). A forest plot of the per-cohort balanced-
     accuracy difference (DA - baseline) with the paired p-value and win/loss
     record makes clear whether the adaptation machinery actually transfers.

Outputs (to results/figures/current_results/ by default):
    loso_strict_da_vs_baseline.{pdf,png}
    loso_strict_da_vs_baseline.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_current_results import model_label  # noqa: E402
from plot_model_significance import nemenyi_cd, plot_cd  # noqa: E402

# Domain-adaptation / invariance variant -> its plain backbone. Each variant
# adds a cross-cohort mechanism (DIVA domain-invariant latent, PhyloDIVA +
# phylogeny, TAXI taxonomy alignment, CORAL second-order feature alignment) on
# top of an otherwise-identical backbone, so the paired difference isolates the
# adaptation's effect on transfer.
DA_PAIRS = [
    ("beta-vae", "diva-beta-vae", "DIVA"),
    ("tree-dtm-vae", "diva-tree-dtm-vae", "DIVA"),
    ("tree-dtm-vae", "phylodiva-tree-dtm-vae", "PhyloDIVA"),
    ("tree-dtm-vae", "taxi-tree-dtm-vae", "TAXI"),
    ("hyp-philrvae", "diva-hyp-philr-nb", "DIVA"),
    ("hyp-philrvae", "phylodiva-hyp-philr-nb", "PhyloDIVA"),
    ("hyp-philrvae", "taxi-hyp-philrvae", "TAXI"),
    ("xgb-baseline", "xgb-coral", "CORAL"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--loso",
        type=Path,
        default=Path("results/figures/current_results/latest_loso_strict_summary.tsv"),
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=Path("results/figures/current_results/latest_single_meta_summary.tsv"),
        help="Single-study summary, for the within-study vs LOSO gap test.",
    )
    p.add_argument("--metric", default="balanced_accuracy", choices=["balanced_accuracy", "f1_macro", "auroc"])
    p.add_argument("--outdir", type=Path, default=Path("results/figures/current_results"))
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return p.parse_args()


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def paired_da_table(piv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for base, da, family in DA_PAIRS:
        if base not in piv.columns or da not in piv.columns:
            continue
        b = piv[base]
        d = piv[da]
        diff = (d - b).dropna()
        n = len(diff)
        try:
            _, p = stats.wilcoxon(d.loc[diff.index], b.loc[diff.index])
        except ValueError:
            p = np.nan
        sd = diff.std(ddof=1)
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        rows.append(
            {
                "baseline": base,
                "da_variant": da,
                "da_family": family,
                "label": f"{model_label(da)}  vs  {model_label(base)}",
                "n_studies": n,
                "delta_mean": float(diff.mean()),
                "delta_ci95": float(1.96 * sem) if np.isfinite(sem) else np.nan,
                "wins": int((diff > 0).sum()),
                "losses": int((diff < 0).sum()),
                "wilcoxon_p": float(p),
            }
        )
    return pd.DataFrame(rows)


def plot_forest(piv: pd.DataFrame, table: pd.DataFrame, metric: str, outdir: Path, formats) -> list[Path]:
    import matplotlib.pyplot as plt

    t = table.sort_values("delta_mean").reset_index(drop=True)
    y = np.arange(len(t))

    def color(row):
        if not np.isfinite(row["wilcoxon_p"]):
            return "#9ca3af"
        if row["wilcoxon_p"] < 0.05:
            return "#117733" if row["delta_mean"] > 0 else "#CC6677"
        return "#9ca3af"

    colors = [color(r) for _, r in t.iterrows()]

    fig, ax = plt.subplots(figsize=(11.5, 0.55 * len(t) + 2.2), constrained_layout=True)

    # per-study points (the raw paired differences) as jitter
    rng = np.random.default_rng(0)
    for i, (_, r) in enumerate(t.iterrows()):
        diff = (piv[r["da_variant"]] - piv[r["baseline"]]).dropna().to_numpy()
        ax.scatter(diff, np.full_like(diff, i) + rng.uniform(-0.13, 0.13, len(diff)),
                   s=22, color=colors[i], alpha=0.35, zorder=2, edgecolor="none")

    ax.hlines(y, t["delta_mean"] - t["delta_ci95"], t["delta_mean"] + t["delta_ci95"],
              color=colors, lw=2.4, zorder=3)
    ax.scatter(t["delta_mean"], y, s=95, color=colors, edgecolor="white", linewidth=1.0, zorder=4)

    ax.axvline(0, ls="--", lw=1.3, color="#374151", zorder=1)
    ax.text(0, len(t) - 0.3, "no effect", fontsize=8.5, color="#6b7280", ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(t["label"], fontsize=9.5)
    ax.set_xlabel(f"Δ {metric.replace('_', ' ')}  (domain-adaptation − baseline, across {int(t['n_studies'].iloc[0])} held-out cohorts)")
    ax.set_title("Does domain adaptation improve strict-LOSO generalisation? (paired Wilcoxon)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_axisbelow(True)

    xmax = float(max(0.06, (t["delta_mean"].abs() + t["delta_ci95"].fillna(0)).max() * 1.15))
    ax.set_xlim(-xmax, xmax)
    for i, (_, r) in enumerate(t.iterrows()):
        ptxt = "p<0.001" if r["wilcoxon_p"] < 1e-3 else f"p={r['wilcoxon_p']:.3f}"
        ax.text(
            xmax * 0.98, i,
            f"{r['delta_mean']:+.3f}  ·  {ptxt}  ·  {r['wins']}W/{r['losses']}L",
            ha="right", va="center", fontsize=8.5, color="#374151",
        )

    # interpretation banner
    ax.text(
        0.5, -0.14,
        "Green = significant gain · Red = significant loss · Grey = n.s. (no transfer benefit)",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#6b7280", style="italic",
    )

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = outdir / f"loso_strict_da_vs_baseline.{fmt}"
        fig.savefig(path, dpi=600 if fmt == "png" else 1200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def model_cd(piv: pd.DataFrame, metric: str, outdir: Path, formats) -> list[Path]:
    """Friedman omnibus + Nemenyi critical-difference diagram across LOSO models."""
    complete = piv.dropna(axis=1)
    ranks = complete.rank(axis=1, ascending=False, method="average")
    avg_rank = ranks.mean(axis=0).sort_values()
    chi2, p = stats.friedmanchisquare(*[complete[c].to_numpy() for c in complete.columns])
    k, n = complete.shape[1], complete.shape[0]
    cd = nemenyi_cd(k, n)
    info = {"friedman_chi2": float(chi2), "friedman_p": float(p), "k": k, "N": n}
    return plot_cd(avg_rank, cd, info, metric, outdir, formats,
                   stem="loso_strict_model_cd", unit="held-out cohorts")


def _boot_ci(vals: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _draw_chance(ax, piv: pd.DataFrame, ac: pd.DataFrame, metric: str, letter: str | None = None) -> None:
    """Per-model mean metric + bootstrap CI vs chance, ordered best->worst."""
    md = ac.sort_values("mean").reset_index(drop=True)
    yA = np.arange(len(md))
    cis = [_boot_ci(piv[m].dropna().to_numpy()) for m in md["model"]]
    lo = np.array([c[0] for c in cis])
    hi = np.array([c[1] for c in cis])
    sig = md["wilcoxon_p_gt_0.5"].to_numpy() < 0.05
    colors = ["#117733" if s else "#6b7280" for s in sig]
    ax.hlines(yA, lo, hi, color=colors, lw=2.4, zorder=2)
    ax.scatter(md["mean"], yA, s=80, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    ax.axvline(0.5, ls="--", lw=1.4, color="#b91c1c", zorder=1)
    ax.text(0.5, len(md) - 0.3, "chance", color="#b91c1c", fontsize=9, ha="center", va="bottom")
    ax.set_yticks(yA)
    ax.set_yticklabels([model_label(m) for m in md["model"]], fontsize=9.5)
    ax.set_xlabel(f"Mean {metric.replace('_', ' ')} across held-out cohorts (95% bootstrap CI)")
    ax.set_title(_titled(letter, "Does any model beat chance under strict LOSO?"), loc="left")
    for i, (_, r) in enumerate(md.iterrows()):
        star = " *" if r["wilcoxon_p_gt_0.5"] < 0.05 else ""
        ax.text(hi[i] + 0.006, i, f"p={r['wilcoxon_p_gt_0.5']:.2f}{star}", va="center", ha="left", fontsize=8, color="#374151")
    ax.set_xlim(0.30, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_axisbelow(True)


def _draw_difficulty(ax, piv: pd.DataFrame, metric: str, letter: str | None = None) -> None:
    """Held-out cohort difficulty (mean across models + bootstrap CI)."""
    cohort_mean = piv.mean(axis=1).sort_values()
    yB = np.arange(len(cohort_mean))
    bcis = [_boot_ci(piv.loc[s].dropna().to_numpy()) for s in cohort_mean.index]
    blo = np.array([c[0] for c in bcis])
    bhi = np.array([c[1] for c in bcis])
    ax.hlines(yB, blo, bhi, color="#2563eb", lw=2.4, zorder=2)
    ax.scatter(cohort_mean.values, yB, s=80, color="#2563eb", edgecolor="white", linewidth=1.0, zorder=3)
    ax.axvline(0.5, ls="--", lw=1.4, color="#b91c1c", zorder=1)
    ax.text(0.5, len(cohort_mean) - 0.3, "chance", color="#b91c1c", fontsize=9, ha="center", va="bottom")
    ax.set_yticks(yB)
    ax.set_yticklabels([s.replace("_", " ") for s in cohort_mean.index], fontsize=9.5)
    ax.set_xlabel(f"Mean {metric.replace('_', ' ')} across models (95% bootstrap CI)")
    ax.set_title(_titled(letter, "Which held-out CRC cohorts are hardest to transfer to?"), loc="left")
    ax.set_xlim(0.30, 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_axisbelow(True)


def plot_chance_and_difficulty(piv: pd.DataFrame, ac: pd.DataFrame, metric: str, outdir: Path, formats) -> list[Path]:
    """A: per-model mean ± bootstrap CI vs chance. B: per-cohort difficulty.

    Writes the combined multipanel plus standalone single-panel versions of
    each panel into ``loso_strict_chance_difficulty_panels/``.
    """
    import matplotlib.pyplot as plt

    # ----- combined multipanel ----------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True, width_ratios=[1.0, 1.0])
    _draw_chance(axA, piv, ac, metric, "A")
    _draw_difficulty(axB, piv, metric, "B")
    paths = _save_fig(fig, outdir, "loso_strict_chance_difficulty", formats)

    # ----- standalone single-panel figures (no panel letters) ---------------
    panel_dir = outdir / "loso_strict_chance_difficulty_panels"
    panels = [
        ("a", "above_chance", (9.0, 7.0), lambda ax: _draw_chance(ax, piv, ac, metric)),
        ("b", "cohort_difficulty", (9.0, 7.0), lambda ax: _draw_difficulty(ax, piv, metric)),
    ]
    for letter, slug, figsize, draw in panels:
        sfig, sax = plt.subplots(figsize=figsize, constrained_layout=True)
        draw(sax)
        paths += _save_fig(sfig, panel_dir, f"loso_strict_chance_difficulty_panel_{letter}_{slug}", formats)
    return paths


# Single-study model_key -> strict-LOSO model name, for the gap comparison.
MODEL_KEY_ALIAS = {"xgboost-baseline": "xgb-baseline"}


def gap_table(single: pd.DataFrame, loso: pd.DataFrame, metric: str):
    single = single.copy()
    single[metric] = pd.to_numeric(single[metric], errors="coerce")
    single["mk"] = single["model_key"].replace(MODEL_KEY_ALIAS)
    common = sorted(set(single["mk"]) & set(loso["model"]))
    studies = sorted(set(single["study"]) & set(loso["held_out_study"]))
    rows = []
    for m in common:
        for s in studies:
            a = single[(single["mk"] == m) & (single["study"] == s)][metric]
            b = loso[(loso["model"] == m) & (loso["held_out_study"] == s)][metric]
            if len(a) and len(b) and np.isfinite(a.iloc[0]) and np.isfinite(b.iloc[0]):
                rows.append({"model": m, "study": s, "within": float(a.iloc[0]), "loso": float(b.iloc[0])})
    g = pd.DataFrame(rows)
    g["gap"] = g["within"] - g["loso"]
    return g, common


def _draw_gap_scatter(ax, g: pd.DataFrame, models: list, letter: str | None = None) -> None:
    """Within-study vs strict-LOSO scatter with the y=x no-gap reference."""
    from plot_current_results import model_color

    for m in models:
        sub = g[g["model"] == m]
        ax.scatter(sub["within"], sub["loso"], s=58, color=model_color(m), edgecolor="white",
                   linewidth=0.8, label=model_label(m), zorder=3)
    lims = [0.25, 1.0]
    ax.plot(lims, lims, ls="--", color="#374151", lw=1.3, zorder=1)
    ax.text(0.97, 0.93, "y = x (no gap)", transform=ax.transAxes, ha="right", fontsize=9, color="#6b7280")
    ax.axhline(0.5, ls=":", lw=1.0, color="#cbd5e1"); ax.axvline(0.5, ls=":", lw=1.0, color="#cbd5e1")
    _, p_all = stats.wilcoxon(g["within"], g["loso"])
    ax.annotate(
        f"Paired Wilcoxon (within vs LOSO): p = {p_all:.2f}\nmean within = {g['within'].mean():.3f}, "
        f"mean LOSO = {g['loso'].mean():.3f}\n→ no significant transfer gap",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left", fontsize=9, color="#374151",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e5e7eb"),
    )
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("Within-study balanced accuracy")
    ax.set_ylabel("Strict-LOSO balanced accuracy")
    ax.set_title(_titled(letter, "Within-study vs cross-cohort (transfer gap)"), loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35); ax.set_axisbelow(True)


def _draw_gap_bars(ax, g: pd.DataFrame, models: list, letter: str | None = None) -> None:
    """Per-model within-study − LOSO gap with paired Wilcoxon p-values."""
    rows = []
    for m in models:
        sub = g[g["model"] == m]
        try:
            _, pm = stats.wilcoxon(sub["within"], sub["loso"])
        except ValueError:
            pm = np.nan
        sem = sub["gap"].std(ddof=1) / np.sqrt(len(sub))
        rows.append({"model": m, "gap": sub["gap"].mean(), "ci": 1.96 * sem, "p": pm})
    gp = pd.DataFrame(rows).sort_values("gap")
    y = np.arange(len(gp))
    ax.hlines(y, gp["gap"] - gp["ci"], gp["gap"] + gp["ci"], color="#6b7280", lw=2.4, zorder=2)
    ax.scatter(gp["gap"], y, s=85, color="#6b7280", edgecolor="white", linewidth=1.0, zorder=3)
    ax.axvline(0, ls="--", lw=1.3, color="#374151")
    ax.set_yticks(y); ax.set_yticklabels([model_label(m) for m in gp["model"]], fontsize=9.5)
    for i, (_, r) in enumerate(gp.iterrows()):
        ax.text(0.98, i + 0.30, f"gap={r['gap']:+.3f}, p={r['p']:.2f}", transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=8.0, color="#374151")
    ax.set_xlabel("Within-study − LOSO balanced accuracy (>0 = worse transfer)")
    ax.set_title(_titled(letter, "Per-model transfer gap (paired Wilcoxon)"), loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.35); ax.set_axisbelow(True)


def plot_gap(g: pd.DataFrame, models: list, metric: str, outdir: Path, formats) -> list[Path]:
    """Within-study vs strict-LOSO transfer gap.

    Writes the combined multipanel plus standalone single-panel versions of
    each panel into ``loso_strict_transfer_gap_panels/``.
    """
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # ----- combined multipanel ----------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 6.8), constrained_layout=True, width_ratios=[1.0, 1.0])
    _draw_gap_scatter(axA, g, models, "A")
    _draw_gap_bars(axB, g, models, "B")
    paths = _save_fig(fig, outdir, "loso_strict_transfer_gap", formats)

    # ----- standalone single-panel figures (no panel letters) ---------------
    panel_dir = outdir / "loso_strict_transfer_gap_panels"
    panels = [
        ("a", "within_vs_loso", (8.5, 7.5), lambda ax: _draw_gap_scatter(ax, g, models)),
        ("b", "per_model_gap", (8.5, 6.5), lambda ax: _draw_gap_bars(ax, g, models)),
    ]
    for letter, slug, figsize, draw in panels:
        sfig, sax = plt.subplots(figsize=figsize, constrained_layout=True)
        draw(sax)
        paths += _save_fig(sfig, panel_dir, f"loso_strict_transfer_gap_panel_{letter}_{slug}", formats)
    return paths


def capda_single_vs_loso(single: pd.DataFrame, loso: pd.DataFrame, metric: str, model: str = "capda-vae"):
    """Pair a model's within-study (single) vs cross-cohort (strict-LOSO) score
    on the held-out CRC cohorts. For CAPDA the single-study form is inert and
    the LOSO form is domain-active, so this isolates the domain setting."""
    single = single.copy()
    single[metric] = pd.to_numeric(single[metric], errors="coerce")
    crc = sorted(set(single["study"]) & set(loso["held_out_study"]))
    s = (single[(single["model_key"] == model) & (single["study"].isin(crc))]
         [["study", metric]].rename(columns={metric: "single"}))
    l = (loso[(loso["model"] == model) & (loso["held_out_study"].isin(crc))]
         [["held_out_study", metric]].rename(columns={"held_out_study": "study", metric: "loso"}))
    m = s.merge(l, on="study")
    m["drop"] = m["single"] - m["loso"]
    return m.sort_values("single", ascending=True).reset_index(drop=True)


def plot_capda_single_vs_loso(m: pd.DataFrame, metric: str, outdir: Path, formats, model: str = "capda-vae") -> list[Path]:
    import matplotlib.pyplot as plt

    y = np.arange(len(m))
    c_single, c_loso = "#0891b2", "#be185d"
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(m) + 2.2), constrained_layout=True)

    ax.hlines(y, m["loso"], m["single"], color="#cbd5e1", lw=2.6, zorder=1)
    ax.scatter(m["single"], y, s=95, color=c_single, edgecolor="white", linewidth=1.0, zorder=3, label="Within-study (single)")
    ax.scatter(m["loso"], y, s=95, color=c_loso, edgecolor="white", linewidth=1.0, zorder=3, label="Strict-LOSO (cross-cohort)")
    ax.axvline(0.5, ls=":", lw=1.3, color="#9ca3af", zorder=0)
    ax.text(0.5, len(m) - 0.3, "chance", fontsize=8.5, color="#6b7280", ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in m["study"]], fontsize=9.5)
    ax.set_xlabel(f"{metric.replace('_', ' ').capitalize()} on held-out CRC cohort")
    ax.set_xlim(0.25, 1.02)
    ax.set_title(f"{model_label(model)}: within-study vs cross-cohort (strict LOSO)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    _, p = stats.wilcoxon(m["single"], m["loso"])
    ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.2f}"
    verdict = "no significant transfer penalty" if p >= 0.05 else "significant transfer drop"
    ax.annotate(
        f"mean within = {m['single'].mean():.3f}   mean LOSO = {m['loso'].mean():.3f}\n"
        f"mean drop = {m['drop'].mean():+.3f}   paired Wilcoxon {ptxt}\n→ {verdict}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left", fontsize=9.5, color="#374151",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e5e7eb"),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = outdir / f"loso_strict_capda_single_vs_loso.{fmt}"
        fig.savefig(path, dpi=600 if fmt == "png" else 1200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def above_chance(piv: pd.DataFrame) -> pd.DataFrame:
    """One-sided Wilcoxon that each model's per-cohort metric exceeds 0.5."""
    rows = []
    for m in piv.columns:
        vals = piv[m].dropna().to_numpy()
        try:
            _, p = stats.wilcoxon(vals - 0.5, alternative="greater")
        except ValueError:
            p = np.nan
        rows.append({"model": m, "model_label": model_label(m), "mean": float(vals.mean()),
                     "median": float(np.median(vals)), "n": len(vals), "wilcoxon_p_gt_0.5": float(p)})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    df = pd.read_csv(args.loso, sep="\t")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    piv = df.pivot_table(index="held_out_study", columns="model", values=args.metric)

    table = paired_da_table(piv)
    table.to_csv(args.outdir / "loso_strict_da_vs_baseline.tsv", sep="\t", index=False)
    ac = above_chance(piv)
    ac.to_csv(args.outdir / "loso_strict_above_chance.tsv", sep="\t", index=False)

    paths = plot_forest(piv, table, args.metric, args.outdir, args.formats)
    paths += model_cd(piv, args.metric, args.outdir, args.formats)
    paths += plot_chance_and_difficulty(piv, ac, args.metric, args.outdir, args.formats)
    if args.meta.exists():
        single = pd.read_csv(args.meta, sep="\t")
        g, gmodels = gap_table(single, df, args.metric)
        g.to_csv(args.outdir / "loso_strict_transfer_gap.tsv", sep="\t", index=False)
        if len(g):
            paths += plot_gap(g, gmodels, args.metric, args.outdir, args.formats)
        # CAPDA-VAE specifically: single-study (inert) vs LOSO (domain-active).
        cap = capda_single_vs_loso(single, df, args.metric, model="capda-vae")
        if len(cap):
            cap.to_csv(args.outdir / "loso_strict_capda_single_vs_loso.tsv", sep="\t", index=False)
            paths += plot_capda_single_vs_loso(cap, args.metric, args.outdir, args.formats)

    print(table.to_string(index=False))
    print("\nAbove-chance (Wilcoxon metric > 0.5):")
    print(ac.to_string(index=False))
    print("\nWrote:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
