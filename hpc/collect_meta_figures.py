#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════
#  collect_meta_figures.py – gather every study's ``figures/`` folder
#
#  Walks a meta-pipeline ``output_root`` (the one produced by
#  ``workflow/Snakefile.meta``) and copies each
#  ``<output_root>/<study>/figures/`` sub-tree into a flat destination
#  directory, renaming each copy to ``figures_<study>/`` so all studies'
#  figures live side-by-side under one location.
#
#  Usage:
#      python hpc/collect_meta_figures.py <output_root> <dest>
#      python hpc/collect_meta_figures.py <output_root> <dest> --dry-run
#      python hpc/collect_meta_figures.py <output_root> <dest> --force
#      python hpc/collect_meta_figures.py <output_root> <dest> --symlink
# ══════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def iter_studies(output_root: Path):
    """Yield (study_name, figures_dir) for every study under *output_root*."""
    for study_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        # Same heuristic as check_meta_logs.py: a study has a models/ subtree.
        if study_dir.name == "logs":
            continue
        if not (study_dir / "models").is_dir():
            continue
        figures_dir = study_dir / "figures"
        if not figures_dir.is_dir():
            yield study_dir.name, None
            continue
        yield study_dir.name, figures_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy every study's figures/ folder from a biomevae meta "
            "output_root into a single destination directory, renaming "
            "each copy to figures_<study>/."
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
        help="Destination directory for the collected figures_<study>/ folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing figures_<study>/ folders in <dest>.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help=(
            "Create symlinks to each study's figures/ folder instead of "
            "copying (much faster; destination follows the original)."
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

    copied = 0
    skipped_missing: list[str] = []
    skipped_exists: list[str] = []
    for study, figures_dir in iter_studies(output_root):
        if figures_dir is None:
            skipped_missing.append(study)
            continue

        target = dest / f"figures_{study}"
        action = "symlink" if args.symlink else "copy"

        if target.exists() or target.is_symlink():
            if not args.force:
                skipped_exists.append(study)
                continue
            if args.dry_run:
                print(f"[dry-run] would remove existing {target}")
            else:
                if target.is_symlink() or target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)

        if args.dry_run:
            print(f"[dry-run] {action} {figures_dir} -> {target}")
            copied += 1
            continue

        if args.symlink:
            target.symlink_to(figures_dir, target_is_directory=True)
        else:
            shutil.copytree(figures_dir, target)
        print(f"{action}: {figures_dir} -> {target}")
        copied += 1

    print(
        f"\nDone: {copied} study figure folder(s) "
        f"{'would be ' if args.dry_run else ''}"
        f"{'linked' if args.symlink else 'copied'} to {dest}."
    )
    if skipped_missing:
        print(
            f"Skipped (no figures/ produced): {len(skipped_missing)} – "
            f"{', '.join(skipped_missing)}"
        )
    if skipped_exists:
        print(
            f"Skipped (destination exists, rerun with --force): "
            f"{len(skipped_exists)} – {', '.join(skipped_exists)}"
        )

    return 0 if copied > 0 or not (skipped_missing or skipped_exists) else 1


if __name__ == "__main__":
    sys.exit(main())
