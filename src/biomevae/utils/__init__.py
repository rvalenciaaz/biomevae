"""Reproducibility utilities shared by :mod:`biomevae`.

The modules under :mod:`biomevae.utils` centralise the helpers used to
make every evaluation in the repository reproducible across runs and
machines:

* :mod:`biomevae.utils.seeding` – ``set_global_seed`` / ``seed_worker``
  helpers that seed ``random``, ``numpy``, ``torch`` (CPU + CUDA) and
  ``PYTHONHASHSEED`` in one call and enable PyTorch's deterministic
  cuDNN / kernel modes.
* :mod:`biomevae.utils.provenance` – ``capture_provenance`` that builds
  a small JSON-serialisable record containing the git SHA, relevant
  package versions, platform, thread counts and a timestamp.  Each
  evaluation result embeds this record under ``metadata['provenance']``
  so that downstream figures / tables can be traced back to the exact
  environment that produced them.
"""

from .seeding import SeededRngs, seed_worker, set_global_seed, temporary_seed
from .provenance import capture_provenance

__all__ = [
    "SeededRngs",
    "capture_provenance",
    "seed_worker",
    "set_global_seed",
    "temporary_seed",
]
