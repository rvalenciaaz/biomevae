"""Lightweight provenance capture for evaluation results.

Every :class:`biomevae.classify.ClassificationResult` and
:class:`biomevae.reconstruction.CrossValResult` embeds a small
JSON-serialisable dictionary describing the exact environment that
produced the numbers.  This makes it possible to reproduce a figure
months later even if the repository has moved on or the host machine
has been reimaged.

The captured fields are deliberately conservative: we avoid anything
that might leak secrets (``HOME``, ``USER``, full environment) and only
record values that are stable across runs of the same commit on the
same host.  When a value cannot be determined (e.g. ``git`` is not on
the ``PATH`` or a package is not installed) the function records
``None`` rather than raising.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

__all__ = ["capture_provenance"]


_DEFAULT_PACKAGES: tuple[str, ...] = (
    "biomevae",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "xgboost",
)


def _safe_pkg_version(name: str) -> Optional[str]:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def _resolve_git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    """Return the current git commit SHA or ``None`` if unavailable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _resolve_git_dirty(cwd: Optional[Path] = None) -> Optional[bool]:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def _resolve_torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch
    except ImportError:
        return info
    info["available"] = True
    info["version"] = getattr(torch, "__version__", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn_version"] = (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        )
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - defensive
            info["device_name"] = None
    info["deterministic_cudnn"] = bool(
        getattr(getattr(torch.backends, "cudnn", None), "deterministic", False)
    )
    info["benchmark_cudnn"] = bool(
        getattr(getattr(torch.backends, "cudnn", None), "benchmark", False)
    )
    return info


def _resolve_thread_counts() -> Dict[str, Any]:
    threads: Dict[str, Any] = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import torch

        threads["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        threads["torch_num_threads"] = None
    return threads


def capture_provenance(
    *,
    seeds: Optional[Iterable[int]] = None,
    packages: Optional[Iterable[str]] = None,
    cwd: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a JSON-serialisable dict describing the current run.

    Parameters
    ----------
    seeds:
        Optional iterable of integer seeds the caller is about to use.
        Recorded verbatim under ``provenance['seeds']`` so that downstream
        consumers can see "this result pooled seeds [42, 43, 44, 45, 46]"
        without digging into ``metadata['seeds']``.
    packages:
        Optional iterable of package names whose versions should be
        recorded.  Defaults to the core scientific stack used by
        :mod:`biomevae`.
    cwd:
        Working directory to run ``git`` in.  Defaults to the current
        process CWD, which is normally the repository root when called
        from a CLI entry point.
    extra:
        Extra key/value pairs to merge into the returned dict.  Useful
        for recording CLI arguments or hyper-parameters alongside the
        provenance block.
    """
    pkg_versions: Dict[str, Optional[str]] = {}
    for name in packages or _DEFAULT_PACKAGES:
        pkg_versions[name] = _safe_pkg_version(name)

    record: Dict[str, Any] = {
        "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": _resolve_git_sha(cwd),
        "git_dirty": _resolve_git_dirty(cwd),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "packages": pkg_versions,
        "torch": _resolve_torch_info(),
        "threads": _resolve_thread_counts(),
    }
    if seeds is not None:
        record["seeds"] = [int(s) for s in seeds]
    if extra:
        record["extra"] = dict(extra)
    return record
