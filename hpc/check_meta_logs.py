#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════
#  check_meta_logs.py – one-off health check for a meta-pipeline output
#
#  Walks a meta-pipeline ``output_root`` directory (the one produced by
#  ``workflow/Snakefile.meta``) and reports signs of trouble in the
#  per-rule ``*.log`` files each study writes under
#  ``<study>/models/<model>/logs/``, ``<study>/figures/logs/`` and
#  ``<study>/models/aggregate/logs/``, plus the top-level
#  ``logs/meta_summary.log``.
#
#  It also scans ``logs/slurm/*.err`` / ``*.out`` when present (or a
#  separate directory passed via ``--slurm-logs``), and flags studies
#  that never produced ``figures/results_summary.tsv``.
#
#  Intended as a quick "is my meta run OK right now?" probe – run it
#  once, read the summary, exit.  Exits 0 when clean, 1 when anything
#  suspicious is found so it can be wired into CI / watchdogs.
#
#  Usage:
#      python hpc/check_meta_logs.py <output_root>
#      python hpc/check_meta_logs.py <output_root> --verbose
#      python hpc/check_meta_logs.py <output_root> \
#          --slurm-logs /path/to/logs/slurm
# ══════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


# Case-insensitive error markers. Matched as whole words where it matters
# so "error" doesn't trip on "NoError" symbols in a traceback frame.
ERROR_PATTERNS = [
    re.compile(r"\btraceback\b", re.IGNORECASE),
    re.compile(r"\bexception\b", re.IGNORECASE),
    re.compile(r"\bfatal\b", re.IGNORECASE),
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"\bkilled\b", re.IGNORECASE),
    re.compile(r"\baborted\b", re.IGNORECASE),
    re.compile(r"cuda out of memory", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"oom[- ]kill", re.IGNORECASE),
    re.compile(r"segmentation fault", re.IGNORECASE),
    re.compile(r"slurmstepd:\s*error", re.IGNORECASE),
    re.compile(r"DUE TO TIME LIMIT", re.IGNORECASE),
    re.compile(r"MissingOutputException", re.IGNORECASE),
]

# Lines that match an error pattern but are obviously benign – skip them
# so the report stays signal-heavy.
IGNORE_PATTERNS = [
    re.compile(r"\b0 errors?\b", re.IGNORECASE),
    re.compile(r"\bno errors?\b", re.IGNORECASE),
    re.compile(r"error[_-]rate", re.IGNORECASE),
    re.compile(r"errorbar", re.IGNORECASE),
    re.compile(r"std[_-]?err(or)?\b", re.IGNORECASE),
    re.compile(r"raise_for_status|on_failure|on_error", re.IGNORECASE),
]

MAX_SAMPLE_LINES = 3  # per log file, shown in the verbose report


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return (line_no, line) tuples for suspicious lines in *path*."""
    hits: list[tuple[int, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                stripped = line.rstrip("\n")
                if any(p.search(stripped) for p in IGNORE_PATTERNS):
                    continue
                if any(p.search(stripped) for p in ERROR_PATTERNS):
                    hits.append((i, stripped))
    except OSError as exc:
        hits.append((0, f"<could not read log: {exc}>"))
    return hits


def classify_path(output_root: Path, log_path: Path) -> tuple[str, str]:
    """Best-effort (study, stage) bucket for a log file path."""
    try:
        rel = log_path.relative_to(output_root)
    except ValueError:
        return ("<external>", log_path.name)

    parts = rel.parts
    if not parts:
        return ("<root>", log_path.name)

    # Top-level logs (e.g. logs/meta_summary.log, logs/slurm/*).
    if parts[0] == "logs":
        if len(parts) >= 3 and parts[1] == "slurm":
            return ("<slurm>", parts[-1])
        return ("<meta>", parts[-1])

    study = parts[0]
    # <study>/models/<model>/logs/<file> or <study>/models/aggregate/logs/...
    if len(parts) >= 5 and parts[1] == "models":
        return (study, f"{parts[2]}/{parts[-1]}")
    # <study>/figures/logs/<file>
    if len(parts) >= 4 and parts[1] == "figures":
        return (study, f"figures/{parts[-1]}")
    return (study, "/".join(parts[1:]))


def find_logs(output_root: Path, extra_slurm_dir: Path | None) -> list[Path]:
    logs = sorted(output_root.rglob("*.log"))
    # SLURM stdout/err – only the bits that tend to capture failures the
    # snakemake-level logs never saw (e.g. node preemption, OOM-kill).
    slurm_dir = extra_slurm_dir or (output_root / "logs" / "slurm")
    if slurm_dir.is_dir():
        logs.extend(sorted(slurm_dir.glob("*.err")))
        logs.extend(sorted(slurm_dir.glob("*.out")))
    return logs


def find_missing_summaries(output_root: Path) -> list[str]:
    """Studies that have a directory but no ``figures/results_summary.tsv``."""
    missing: list[str] = []
    for study_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        if study_dir.name in {"logs"}:
            continue
        # Heuristic: only treat it as a study if it has a models/ subtree.
        if not (study_dir / "models").is_dir():
            continue
        summary = study_dir / "figures" / "results_summary.tsv"
        if not summary.exists():
            missing.append(study_dir.name)
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a biomevae meta-pipeline output directory for signs of "
            "errors in its per-rule log files."
        ),
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Path to the meta pipeline output_root (as set in the config).",
    )
    parser.add_argument(
        "--slurm-logs",
        type=Path,
        default=None,
        help=(
            "Additional directory of SLURM .out/.err files to scan "
            "(defaults to <output_root>/logs/slurm if present)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print up to %d matching lines per log." % MAX_SAMPLE_LINES,
    )
    args = parser.parse_args(argv)

    output_root: Path = args.output_root.expanduser().resolve()
    if not output_root.is_dir():
        print(f"Error: {output_root} is not a directory.", file=sys.stderr)
        return 2

    logs = find_logs(output_root, args.slurm_logs)
    missing = find_missing_summaries(output_root)

    per_study: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(dict)
    total_hits = 0
    for log_path in logs:
        hits = scan_file(log_path)
        if not hits:
            continue
        study, stage = classify_path(output_root, log_path)
        per_study[study][stage] = hits
        total_hits += len(hits)

    print(f"Checked {len(logs)} log file(s) under {output_root}")
    if not per_study and not missing:
        print("No error markers found. ✔")
        return 0

    if per_study:
        print(
            f"Found {total_hits} suspicious line(s) across "
            f"{sum(len(v) for v in per_study.values())} log file(s):"
        )
        for study in sorted(per_study):
            print(f"\n[{study}]")
            for stage in sorted(per_study[study]):
                hits = per_study[study][stage]
                print(f"  {stage}: {len(hits)} hit(s)")
                if args.verbose:
                    for lineno, text in hits[:MAX_SAMPLE_LINES]:
                        print(f"      L{lineno}: {text}")
                    if len(hits) > MAX_SAMPLE_LINES:
                        print(
                            f"      … and {len(hits) - MAX_SAMPLE_LINES} "
                            "more (rerun without --verbose or open the log)."
                        )

    if missing:
        print(
            f"\nStudies missing figures/results_summary.tsv "
            f"({len(missing)}): {', '.join(missing)}"
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
