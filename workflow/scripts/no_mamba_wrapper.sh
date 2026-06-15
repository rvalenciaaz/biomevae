#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# no_mamba_wrapper.sh – drop-in replacement for ``mamba`` used by the
# Snakemake aggregate rule.
#
# The HPC shell scripts in ``hpc/`` invoke CLIs through a mamba wrapper:
#
#     "${MAMBA_EXEC}" run -n "${CONDA_ENV}" biomevae-cli ARGS
#
# When we reuse those scripts from Snakemake (which is already running
# inside an activated environment) we do not want another layer of
# env activation.  Pointing ``MAMBA_EXEC`` at this script achieves
# exactly that: it strips the ``run -n <env>`` prefix and execs the
# remaining command directly.
#
# Behaviour:
#   run -n ENV  CMD [ARGS…]   → exec CMD [ARGS…]
#   anything else             → exec the given command verbatim so that
#                               accidental calls (``mamba --version``)
#                               do not crash the pipeline.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

if [[ $# -ge 3 && "$1" == "run" && "$2" == "-n" ]]; then
    shift 3
fi

exec "$@"
