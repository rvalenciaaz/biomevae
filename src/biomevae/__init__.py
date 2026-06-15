"""Top-level package for :mod:`biomevae`.

The module exposes the most commonly used subpackages (``data``, ``models``,
``trainers`` …) via lazy imports so that simply importing :mod:`biomevae` keeps the
initial import time small while still providing the canonical attribute-based
access pattern::

    from biomevae import data, models

The package version is resolved from the installed metadata with a sensible
fallback for editable installs.
"""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

__all__ = [
    "__version__",
    "data",
    "losses",
    "loso",
    "taxonomy",
    "models",
    "trainers",
    "optuna_utils",
    "reconstruction",
    "extract",
    "classify",
    "utils",
]

_lazy_submodules = {name for name in __all__ if name != "__version__"}

try:  # pragma: no cover - importlib.metadata available in runtime envs
    __version__ = _pkg_version("biomevae")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.dev"


def __getattr__(name: str) -> Any:
    if name in _lazy_submodules:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
