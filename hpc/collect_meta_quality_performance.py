#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════
#  collect_meta_quality_performance.py – gather per-study reconstruction
#  quality and classification performance plots plus the cross-study
#  meta summary into a single destination directory.
#
#  Walks a meta-pipeline ``output_root`` (the one produced by
#  ``workflow/Snakefile.meta``) and, for every study, copies
#  ``fig2_classification_performance.png`` and
#  ``fig4_reconstruction_quality.png`` from
#  ``<output_root>/<study>/figures/`` directly into ``<dest>/`` with
#  the study name appended to the filename, e.g.
#  ``fig2_classification_performance_<study>.png``.
#  In addition, the cross-study ``<output_root>/meta_summary.tsv``
#  (written above the study folders by the ``meta_summary`` rule)
#  is copied to ``<dest>/meta_summary.tsv``.
#
#  Usage:
#      python hpc/collect_meta_quality_performance.py <output_root> <dest>
#      python hpc/collect_meta_quality_performance.py <output_root> <dest> --dry-run
#      python hpc/collect_meta_quality_performance.py <output_root> <dest> --force
#      python hpc/collect_meta_quality_performance.py <output_root> <dest> --symlink
# ══════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PLOT_BASENAMES = (
    "fig2_classification_performance.png",
    "fig4_reconstruction_quality.png",
)

META_SUMMARY_BASENAME = "meta_summary.tsv"


def iter_studies(output_root: Path):
    """Yield (study_name, figures_dir) for every study under *output_root*."""
    for study_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        # Same heuristic as collect_meta_figures.py / check_meta_logs.py:
        # a study has a models/ subtree.
        if study_dir.name == "logs":
            continue
        if not (study_dir / "models").is_dir():
            continue
        figures_dir = study_dir / "figures"
        if not figures_dir.is_dir():
            yield study_dir.name, None
            continue
        yield study_dir.name, figures_dir


def _place(src: Path, dst: Path, *, symlink: bool, force: bool, dry_run: bool) -> str:
    """Copy/symlink a single file. Returns one of {"placed", "skipped", "missing"}."""
    if not src.is_file():
        return "missing"

    action = "symlink" if symlink else "copy"

    if dst.exists() or dst.is_symlink():
        if not force:
            return "skipped"
        if dry_run:
            print(f"[dry-run] would remove existing {dst}")
        else:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)

    if dry_run:
        print(f"[dry-run] {action} {src} -> {dst}")
        return "placed"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)
    print(f"{action}: {src} -> {dst}")
    return "placed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy each study's reconstruction-quality and classification-"
            "performance plots from a biomevae meta output_root into a "
            "single destination directory, plus the cross-study "
            "meta_summary.tsv that lives above the study folders."
        ),
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Path to the meta pipeline output_root (as set in the config).",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="Destination directory for the collected files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in <dest>.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help=(
            "Create symlinks to each source file instead of copying "
            "(much faster; destination follows the original)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be copied without writing anything.",
    )
    args = parser.parse_args(argv)

    output_root: Path = args.output_root.expanduser().resolve()
    dest: Path = args.dest.expanduser().resolve()

    if not output_root.is_dir():
        print(f"Error: {output_root} is not a directory.", file=sys.stderr)
        return 2
    if dest == output_root or output_root in dest.parents:
        print(
            f"Error: destination {dest} is inside output_root; refusing "
            "to copy to avoid recursive self-inclusion.",
            file=sys.stderr,
        )
        return 2

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    placed = 0
    skipped_no_figures: list[str] = []
    skipped_missing_plots: list[tuple[str, list[str]]] = []
    skipped_exists: list[tuple[str, list[str]]] = []
    for study, figures_dir in iter_studies(output_root):
        if figures_dir is None:
            skipped_no_figures.append(study)
            continue

        study_missing: list[str] = []
        study_exists: list[str] = []
        for basename in PLOT_BASENAMES:
            src = figures_dir / basename
            stem, suffix = basename.rsplit(".", 1)
            dst = dest / f"{stem}_{study}.{suffix}"
            status = _place(
                src, dst,
                symlink=args.symlink,
                force=args.force,
                dry_run=args.dry_run,
            )
            if status == "placed":
                placed += 1
            elif status == "skipped":
                study_exists.append(basename)
            elif status == "missing":
                study_missing.append(basename)

        if study_missing:
            skipped_missing_plots.append((study, study_missing))
        if study_exists:
            skipped_exists.append((study, study_exists))

    # Cross-study meta summary that sits above the study folders.
    meta_src = output_root / META_SUMMARY_BASENAME
    meta_dst = dest / META_SUMMARY_BASENAME
    meta_status = _place(
        meta_src, meta_dst,
        symlink=args.symlink,
        force=args.force,
        dry_run=args.dry_run,
    )
    if meta_status == "placed":
        placed += 1

    verb = "would be " if args.dry_run else ""
    medium = "linked" if args.symlink else "copied"
    print(f"\nDone: {placed} file(s) {verb}{medium} to {dest}.")
    if meta_status == "missing":
        print(
            f"Note: {meta_src} not found – run the meta_summary rule first "
            "to produce it."
        )
    elif meta_status == "skipped":
        print(
            f"Skipped meta_summary.tsv (destination exists, rerun with "
            f"--force)."
        )
    if skipped_no_figures:
        print(
            f"Studies with no figures/ produced: {len(skipped_no_figures)} – "
            f"{', '.join(skipped_no_figures)}"
        )
    if skipped_missing_plots:
        print("Studies missing one or more requested plots:")
        for study, missing in skipped_missing_plots:
            print(f"  {study}: {', '.join(missing)}")
    if skipped_exists:
        print("Files skipped because destination exists (rerun with --force):")
        for study, names in skipped_exists:
            print(f"  {study}: {', '.join(names)}")

    any_skips = (
        skipped_no_figures
        or skipped_missing_plots
        or skipped_exists
        or meta_status in ("missing", "skipped")
    )
    return 0 if placed > 0 or not any_skips else 1


if __name__ == "__main__":
    sys.exit(main())
