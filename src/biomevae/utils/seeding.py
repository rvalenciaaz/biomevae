"""Global seeding and RNG state helpers for reproducible evaluations.

Setting a single seed with ``numpy.random.seed(seed)`` or
``torch.manual_seed(seed)`` is not enough to make a PyTorch / sklearn /
XGBoost pipeline deterministic.  A reproducible run also needs:

* ``PYTHONHASHSEED`` set *before* any hash-based container is populated,
* Python's built-in :mod:`random` seeded (used by libraries like
  ``imbalanced-learn``),
* NumPy's legacy global state (used by sklearn estimators that still
  accept ``random_state=None``),
* PyTorch's CPU *and* every CUDA device state,
* cuDNN put into deterministic mode with benchmarking disabled,
* ``torch.use_deterministic_algorithms`` flipped on when the caller is
  willing to take the throughput hit,
* the ``CUBLAS_WORKSPACE_CONFIG`` environment variable, which cuBLAS
  requires at ``use_deterministic_algorithms`` time.

:func:`set_global_seed` takes care of all of the above and returns a
:class:`SeededRngs` bundle containing Python/NumPy ``Random`` generators
seeded from the same master seed, so callers that need multiple
independent streams can derive them without touching the globals again.

The helpers are intentionally safe to call without torch installed
(imports are wrapped in ``try`` so that pure-numpy code paths work on
environments without CUDA or PyTorch).
"""

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import numpy as np


@dataclass
class SeededRngs:
    """Bundle of independent RNGs derived from a master seed.

    Attributes
    ----------
    seed:
        The master seed that was used.
    python_random:
        :class:`random.Random` instance seeded with ``seed``.
    numpy_rng:
        :class:`numpy.random.Generator` instance seeded with ``seed``
        (via ``numpy.random.default_rng``).
    """

    seed: int
    python_random: random.Random
    numpy_rng: np.random.Generator


def _resolve_deterministic_torch() -> bool:
    """Respect ``BIOMEVAE_DETERMINISTIC_TORCH`` env var (default: on)."""
    value = os.environ.get("BIOMEVAE_DETERMINISTIC_TORCH", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def set_global_seed(
    seed: int,
    *,
    deterministic_torch: Optional[bool] = None,
    strict: bool = False,
) -> SeededRngs:
    """Seed every RNG used by :mod:`biomevae` and return an RNG bundle.

    Parameters
    ----------
    seed:
        Master seed; will be cast to ``int`` and normalised modulo
        ``2**32`` so that NumPy accepts it.
    deterministic_torch:
        If ``True`` (default) enable cuDNN deterministic mode and
        disable its auto-tuner.  Set to ``False`` to preserve
        throughput at the cost of run-to-run variance.  When ``None``
        the default is read from the ``BIOMEVAE_DETERMINISTIC_TORCH``
        environment variable.
    strict:
        When ``True`` also flip ``torch.use_deterministic_algorithms``
        on.  This raises if any operation in the call stack falls back
        to a non-deterministic kernel.  Defaults to ``False`` because a
        number of sparse matmul / scatter kernels used by
        :mod:`biomevae.models` are not covered by the deterministic
        registry in recent PyTorch releases.

    Returns
    -------
    SeededRngs
        A bundle containing the master seed along with a
        :class:`random.Random` and a :class:`numpy.random.Generator`
        seeded from the same value.  Downstream code that needs an
        independent stream (e.g. per-fold splits) should derive it from
        ``rngs.numpy_rng`` rather than calling :func:`set_global_seed`
        again.
    """

    normalised = int(seed) % (2**32)
    if deterministic_torch is None:
        deterministic_torch = _resolve_deterministic_torch()

    os.environ.setdefault("PYTHONHASHSEED", str(normalised))
    os.environ["PYTHONHASHSEED"] = str(normalised)
    # cuBLAS requires this env var to enable determinism.  ``:4096:8``
    # is the value recommended by the PyTorch docs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(normalised)
    np.random.seed(normalised)

    try:  # pragma: no cover - torch is optional at import time
        import torch

        torch.manual_seed(normalised)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalised)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if strict:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch versions lack ``warn_only``.
                torch.use_deterministic_algorithms(True)
    except ImportError:  # pragma: no cover - pure-numpy environments
        pass

    return SeededRngs(
        seed=normalised,
        python_random=random.Random(normalised),
        numpy_rng=np.random.default_rng(normalised),
    )


def _snapshot_rng_state() -> Dict[str, Any]:
    """Capture the current RNG state for every RNG touched by :func:`set_global_seed`."""
    snapshot: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_legacy": np.random.get_state(),
    }
    try:  # pragma: no cover - torch optional
        import torch

        snapshot["torch_cpu"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            snapshot["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:  # pragma: no cover - pure-numpy environments
        pass
    return snapshot


def _restore_rng_state(snapshot: Dict[str, Any]) -> None:
    """Restore every RNG state captured by :func:`_snapshot_rng_state`."""
    if "python_random" in snapshot:
        random.setstate(snapshot["python_random"])
    if "numpy_legacy" in snapshot:
        np.random.set_state(snapshot["numpy_legacy"])
    try:  # pragma: no cover - torch optional
        import torch

        if "torch_cpu" in snapshot:
            torch.random.set_rng_state(snapshot["torch_cpu"])
        if "torch_cuda" in snapshot and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(snapshot["torch_cuda"])
    except ImportError:  # pragma: no cover - pure-numpy environments
        pass


@contextlib.contextmanager
def temporary_seed(
    seed: int,
    *,
    deterministic_torch: Optional[bool] = None,
    strict: bool = False,
) -> Iterator[SeededRngs]:
    """Context manager that seeds every RNG and restores state on exit.

    Equivalent to calling :func:`set_global_seed` inside a ``try/finally``
    block that snapshots and restores the ``random``, NumPy, and Torch
    (CPU+CUDA) RNG states.  Use this in training loops that run multiple
    independent trials back-to-back (hyper-parameter sweeps, per-fold
    cross validation) so that each trial starts from a reproducible
    state without leaking RNG state into the outer driver loop.
    """
    snapshot = _snapshot_rng_state()
    try:
        yield set_global_seed(
            seed,
            deterministic_torch=deterministic_torch,
            strict=strict,
        )
    finally:
        _restore_rng_state(snapshot)


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` that re-seeds each worker.

    Each PyTorch DataLoader worker is forked with a unique ``base_seed``
    that PyTorch derives from the current generator state.  We use that
    base seed to re-initialise NumPy and Python's ``random`` module
    inside the worker so that augmentations / shuffles are deterministic
    across runs.  The PyTorch RNG is already seeded by DataLoader
    itself, so there is nothing extra to do for it.
    """

    try:  # pragma: no cover - torch optional
        import torch

        worker_seed = torch.initial_seed() % (2**32)
    except ImportError:  # pragma: no cover - pure-numpy environments
        worker_seed = worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
