#!/usr/bin/env python3
"""Illustrate the PhILR (phylogenetic isometric log-ratio) transform.

A minimal three-taxon worked example:

    A = 0.50   (lineage 1, on its own)
    B = 0.25   (lineage 2)
    C = 0.25   (lineage 2)

with tree topology ((B, C), A). PhILR places one isometric log-ratio
"balance" at each internal node of the tree:

    b = sqrt( (r * s) / (r + s) ) * ln( g(num) / g(den) )

where r, s are the tip counts of the two child clades and g(.) is the
geometric mean of the relative abundances in a clade. The figure shows the
tree, the abundances, and the resulting balance coordinates.

Writes a high-resolution PNG and a vector PDF to results/figures/.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


# ---- example composition -------------------------------------------------
ABUND = {"A": 0.50, "B": 0.25, "C": 0.25}

# colours: lineage 1 (A) vs lineage 2 (B, C)
COL_A = "#2563eb"   # blue   - lineage 1
COL_B = "#ea580c"   # orange - lineage 2
COL_C = "#f59e0b"   # amber  - lineage 2
LINEAGE2 = "#fb923c"


def gmean(values: list[float]) -> float:
    prod = 1.0
    for v in values:
        prod *= v
    return prod ** (1.0 / len(values))


def balance(num: list[float], den: list[float]) -> float:
    r, s = len(num), len(den)
    return math.sqrt((r * s) / (r + s)) * math.log(gmean(num) / gmean(den))


def configure() -> None:
    plt.rcParams.update(
        {
            # transparent canvas so the figure drops onto any poster background
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "text.antialiased": True,
            "lines.antialiased": True,
        }
    )


def main() -> None:
    configure()

    b1 = balance([ABUND["A"]], [ABUND["B"], ABUND["C"]])  # {A} vs {B,C}
    b2 = balance([ABUND["B"]], [ABUND["C"]])              # {B} vs {C}

    fig = plt.figure(figsize=(13.5, 8.0))
    # Explicit positions (not a relative hspace) so the info band always sits
    # well clear of the panel x-axis labels regardless of matplotlib version.
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.35, 1.0, 1.05],
        wspace=0.32, left=0.045, right=0.975, top=0.97, bottom=0.42,
    )
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bal = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_axes([0.045, 0.035, 0.93, 0.24])
    ax_info.axis("off")

    # ------------------------------------------------------------------ tree
    ax_tree.set_title("Phylogenetic tree + balances", fontsize=14, fontweight="bold", pad=14)
    # tip / node coordinates
    yA, yB, yC = 2.7, 1.05, 0.25
    x_tip = 2.05
    n2 = (1.25, (yB + yC) / 2)          # parent of B, C
    n1 = (0.45, (yA + n2[1]) / 2)       # root: A vs (B,C)

    edge_kw = dict(color="#334155", lw=2.6, solid_capstyle="round", zorder=1)
    # root n1 -> A and -> n2  (rectangular cladogram)
    ax_tree.plot([n1[0], n1[0]], [n2[1], yA], **edge_kw)
    ax_tree.plot([n1[0], x_tip], [yA, yA], **edge_kw)
    ax_tree.plot([n1[0], n2[0]], [n2[1], n2[1]], **edge_kw)
    # node n2 -> B and -> C
    ax_tree.plot([n2[0], n2[0]], [yC, yB], **edge_kw)
    ax_tree.plot([n2[0], x_tip], [yB, yB], **edge_kw)
    ax_tree.plot([n2[0], x_tip], [yC, yC], **edge_kw)

    # balance nodes
    for (nx, ny), lab in [(n1, "b₁"), (n2, "b₂")]:
        ax_tree.add_patch(Circle((nx, ny), 0.135, facecolor="#0f172a",
                                 edgecolor="white", lw=1.6, zorder=3))
        ax_tree.text(nx, ny, lab, ha="center", va="center", color="white",
                     fontsize=11.5, fontweight="bold", zorder=4)

    # tip markers + labels (abundances are shown in the middle panel)
    tip_info = [("A", yA, COL_A), ("B", yB, COL_B), ("C", yC, COL_C)]
    for name, y, col in tip_info:
        ax_tree.add_patch(Circle((x_tip, y), 0.11, facecolor=col,
                                 edgecolor="white", lw=1.4, zorder=3))
        ax_tree.text(x_tip + 0.20, y, f"{name}", ha="left", va="center",
                     fontsize=14, fontweight="bold", color=col)

    # lineage annotations
    ax_tree.annotate("lineage 1", xy=(1.15, yA), xytext=(1.15, yA + 0.42),
                     ha="center", fontsize=10.5, color=COL_A, fontweight="bold")
    ax_tree.annotate("lineage 2", xy=(1.65, (yB + yC) / 2), xytext=(1.62, yC - 0.42),
                     ha="center", fontsize=10.5, color=COL_B, fontweight="bold")

    ax_tree.set_xlim(0.1, 3.2)
    ax_tree.set_ylim(-0.55, 3.35)
    ax_tree.axis("off")

    # ------------------------------------------------------- composition bars
    ax_bar.set_title("Relative abundance", fontsize=14, fontweight="bold", pad=14)
    names = ["A", "B", "C"]
    cols = [COL_A, COL_B, COL_C]
    ys = [2.7, 1.05, 0.25]
    for name, y, col in zip(names, ys, cols):
        ax_bar.barh(y, ABUND[name], height=0.5, color=col, edgecolor="white", lw=1.0)
        ax_bar.text(ABUND[name] + 0.015, y, f"{ABUND[name]:.2f}", va="center",
                    ha="left", fontsize=12, color="#334155")
        ax_bar.text(-0.02, y, name, va="center", ha="right", fontsize=13,
                    fontweight="bold", color=col)
    ax_bar.set_xlim(0, 0.62)
    ax_bar.set_ylim(-0.55, 3.35)
    ax_bar.set_xlabel("proportion", fontsize=11)
    ax_bar.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax_bar.spines[sp].set_visible(False)
    ax_bar.spines["bottom"].set_color("#334155")
    ax_bar.tick_params(labelsize=10)

    # ----------------------------------------------------------- balance plot
    ax_bal.set_title("ILR balance coordinates", fontsize=14, fontweight="bold", pad=14)
    bvals = [b1, b2]
    blabs = ["b₁  (A | B,C)", "b₂  (B | C)"]
    ypos = [1.0, 0.0]
    bar_cols = ["#0f766e", "#94a3b8"]
    ax_bal.axvline(0, color="#334155", lw=1.1, zorder=1)
    for y, val, col in zip(ypos, bvals, bar_cols):
        ax_bal.barh(y, val, height=0.42, color=col, edgecolor="white", lw=1.0, zorder=2)
        off = 0.018 if val >= 0 else -0.018
        ha = "left" if val >= 0 else "right"
        ax_bal.text(val + off, y, f"{val:+.3f}", va="center", ha=ha,
                    fontsize=12.5, fontweight="bold", color="#0f172a")
    ax_bal.set_yticks(ypos)
    ax_bal.set_yticklabels(blabs, fontsize=12)
    ax_bal.set_xlim(-0.2, 0.78)
    ax_bal.set_ylim(-0.6, 1.6)
    ax_bal.set_xlabel("balance value", fontsize=11)
    for sp in ["top", "right", "left"]:
        ax_bal.spines[sp].set_visible(False)
    ax_bal.spines["bottom"].set_color("#334155")
    ax_bal.grid(True, axis="x", alpha=0.4)
    ax_bal.set_axisbelow(True)

    # arrows between panels (aligned to the vertical centre of the top row)
    y_mid = (ax_bar.get_position().y0 + ax_bar.get_position().y1) / 2
    for (a0, a1) in [(ax_tree, ax_bar), (ax_bar, ax_bal)]:
        arr = FancyArrowPatch(
            (a0.get_position().x1 + 0.004, y_mid),
            (a1.get_position().x0 - 0.004, y_mid),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22,
            lw=2.2, color="#94a3b8",
        )
        fig.add_artist(arr)

    # ----------------------------------------------------- info / formula band
    # left: definition + general formula
    ax_info.text(0.035, 0.80, "ILR balance at each internal node",
                 transform=ax_info.transAxes, fontsize=12.5, fontweight="bold",
                 color="#0f172a", va="center")
    ax_info.text(
        0.035, 0.36,
        r"$b \;=\; \sqrt{\dfrac{r\,s}{r+s}}\;\ln\dfrac{g(\mathrm{num})}{g(\mathrm{den})}$",
        transform=ax_info.transAxes, fontsize=15, color="#0f172a", va="center",
    )
    ax_info.text(0.30, 0.36,
                 "$r,s$: tips per\nchild clade\n$g(\\cdot)$: geometric mean",
                 transform=ax_info.transAxes, fontsize=10, color="#475569",
                 va="center", linespacing=1.5)
    # vertical divider
    ax_info.plot([0.475, 0.475], [0.12, 0.88], transform=ax_info.transAxes,
                 color="#cbd5e1", lw=1.1, zorder=1)
    # right: worked example  (g(B,C)=sqrt(0.25*0.25)=0.25, written compactly)
    ax_info.text(0.515, 0.80, "This example", transform=ax_info.transAxes,
                 fontsize=12.5, fontweight="bold", color="#0f172a", va="center")
    ax_info.text(
        0.515, 0.47,
        r"$b_1 = \sqrt{\frac{2}{3}}\;\ln\dfrac{0.50}{0.25} = +0.566$",
        transform=ax_info.transAxes, fontsize=13, color="#0f766e", va="center",
    )
    ax_info.text(0.80, 0.47, r"$\rightarrow$  A elevated vs (B, C)",
                 transform=ax_info.transAxes, fontsize=11, color="#475569", va="center")
    ax_info.text(
        0.515, 0.16,
        r"$b_2 = \sqrt{\frac{1}{2}}\;\ln\dfrac{0.25}{0.25} = 0.000$",
        transform=ax_info.transAxes, fontsize=13, color="#64748b", va="center",
    )
    ax_info.text(0.80, 0.16, r"$\rightarrow$  B and C balanced",
                 transform=ax_info.transAxes, fontsize=11, color="#475569", va="center")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    outdir = Path("results/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "philr_illustration.png"
    pdf = outdir / "philr_illustration.pdf"
    fig.savefig(png, dpi=1200, bbox_inches="tight", pad_inches=0.12, transparent=True)
    fig.savefig(pdf, dpi=1200, bbox_inches="tight", pad_inches=0.12, transparent=True)
    plt.close(fig)
    print(f"b1 (A | B,C) = {b1:+.4f}")
    print(f"b2 (B | C)   = {b2:+.4f}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
