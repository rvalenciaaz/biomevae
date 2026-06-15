"""Tests for the shared Optuna runner used by every DIVA training CLI.

The runner lives in :mod:`biomevae.cli._diva_common` and brings the LOSO
training stage to the same rigour as the single-study pipeline (Optuna
search + retrain-with-best + per-trial seed variation). These tests
exercise it with a stub ``train_fn`` so they do not require torch /
torch_geometric / geoopt / a real DIVA backbone.

They DO require ``optuna`` (a runtime dependency of the optuna sweep),
but skip cleanly when it is missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")  # _diva_common pulls torch in at import time

from biomevae.cli._diva_common import (  # noqa: E402  - after importorskip
    DEFAULT_DIVA_SEARCH_SPACE,
    add_optuna_cli_args,
    run_diva_optuna,
)


def _make_args(outdir: Path, **overrides: Any) -> argparse.Namespace:
    """Build a minimal argparse namespace mimicking a DIVA CLI."""
    base = dict(
        outdir=str(outdir),
        seed=42,
        # Defaults the runner mutates when a search-space entry overrides them.
        latent_d=4,
        latent_y=8,
        latent_x=8,
        lr=1e-3,
        dropout=0.1,
        alpha_d=1.0,
        alpha_y=10.0,
        beta_max=1.0,
        kl_warmup_frac=0.25,
        free_bits=0.02,
        batch_size=64,
        # Optuna flags
        optuna=True,
        optuna_trials=4,
        optuna_config=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_add_optuna_cli_args_registers_three_flags():
    parser = argparse.ArgumentParser("dummy")
    add_optuna_cli_args(parser)
    args = parser.parse_args(["--optuna", "--optuna-trials", "7"])
    assert args.optuna is True
    assert args.optuna_trials == 7
    assert args.optuna_config is None


def test_default_search_space_keys_match_diva_argparse_attrs():
    """Default search-space keys must map onto argparse attributes shared
    by every DIVA CLI; otherwise overrides would silently no-op.
    """
    expected = {
        "latent_d", "latent_y", "latent_x", "lr", "dropout",
        "alpha_y", "beta_max", "kl_warmup_frac", "free_bits", "batch_size",
    }
    assert expected.issubset(DEFAULT_DIVA_SEARCH_SPACE.keys())


def test_run_diva_optuna_minimises_and_retrains_with_best(tmp_path):
    """End-to-end: 4 trials with deterministic stub objective; the runner
    must (1) call train_fn for each trial, (2) call it once more with the
    best overrides in the user's outdir, (3) write
    ``optuna_best_params.json`` with the winning ``seed`` and overrides.
    """
    args = _make_args(tmp_path, optuna_trials=4)

    calls = []

    def stub_train(a, outdir, *, verbose=True):
        # Record (seed, lr, outdir) so the test can verify per-trial seed
        # variation and the final retrain location.
        calls.append({
            "seed": a.seed,
            "lr": float(a.lr),
            "outdir": str(outdir),
            "verbose": verbose,
        })
        # Lower lr → better val (deterministic so Optuna picks the
        # smallest-lr trial as best).
        return {"best_val": float(a.lr)}

    res = run_diva_optuna(args, stub_train)

    # 4 trials + 1 final retrain.
    assert len(calls) == 5
    # Trial seeds vary monotonically as base_seed + trial.number.
    trial_seeds = [c["seed"] for c in calls[:4]]
    assert trial_seeds == [42, 43, 44, 45]
    # Trial outdirs are each below ``optuna_trials/``; final retrain
    # writes to the user-requested outdir.
    for c in calls[:4]:
        assert "/optuna_trials/trial_" in c["outdir"]
        assert c["verbose"] is False
    assert calls[-1]["outdir"] == str(tmp_path)
    assert calls[-1]["verbose"] is True
    # The final retrain uses the best trial's seed and overrides.
    best_lr = min(c["lr"] for c in calls[:4])
    assert calls[-1]["lr"] == pytest.approx(best_lr)

    # Best-params JSON is written.
    best_path = tmp_path / "optuna_best_params.json"
    assert best_path.exists()
    payload = json.loads(best_path.read_text())
    assert "seed" in payload
    assert payload["seed"] in trial_seeds
    # Trials CSV is best-effort but typically present.
    # (do not assert; optuna versions vary)

    # Returned res is the final retrain's res dict.
    assert res["best_val"] == pytest.approx(best_lr)


def test_run_diva_optuna_rejects_zero_trials(tmp_path):
    args = _make_args(tmp_path, optuna_trials=0)

    def stub_train(a, outdir, *, verbose=True):
        return {"best_val": 1.0}

    with pytest.raises(SystemExit):
        run_diva_optuna(args, stub_train)


def test_run_diva_optuna_uses_user_supplied_search_space(tmp_path):
    """``--optuna-config`` must override the default DIVA search space."""
    cfg_path = tmp_path / "search.json"
    cfg_path.write_text(json.dumps({
        # A single, deterministic categorical so we can assert the
        # override took effect.
        "latent_y": {"method": "suggest_categorical", "choices": [99]},
    }))

    args = _make_args(tmp_path, optuna_trials=2, optuna_config=str(cfg_path))

    seen_latent_y = []

    def stub_train(a, outdir, *, verbose=True):
        seen_latent_y.append(int(a.latent_y))
        return {"best_val": float(a.lr)}

    run_diva_optuna(args, stub_train)
    # Every trial + the final retrain saw the override.
    assert seen_latent_y == [99, 99, 99]


def test_run_diva_optuna_skips_non_finite_trials(tmp_path):
    """Non-finite ``best_val`` must not crash the search; Optuna should
    still pick a finite-value trial as best.
    """
    args = _make_args(tmp_path, optuna_trials=3)

    def stub_train(a, outdir, *, verbose=True):
        # First trial returns NaN; remaining trials return finite values.
        if a.seed == 42:
            return {"best_val": float("nan")}
        return {"best_val": float(a.lr)}

    res = run_diva_optuna(args, stub_train)
    assert res["best_val"] == res["best_val"]  # finite (not NaN)


def test_run_diva_optuna_aborts_when_every_trial_errors(tmp_path):
    """If every trial raises before recording a value, Optuna's
    ``best_trial`` lookup itself raises.  The runner must convert that
    into a friendly ``SystemExit`` so the Snakemake log points at the
    underlying problem instead of an opaque traceback.
    """
    args = _make_args(tmp_path, optuna_trials=2)

    class _OnlyTrainError(RuntimeError):
        pass

    def stub_train(a, outdir, *, verbose=True):
        raise _OnlyTrainError("synthetic training failure")

    with pytest.raises(SystemExit, match="every trial failed"):
        run_diva_optuna(args, stub_train)


def test_run_diva_optuna_warns_on_unknown_override_key(tmp_path):
    """Typos / wrong-CLI keys must warn rather than silently no-op.

    The single-study CLI uses ``latent_dim``; the DIVA CLIs use
    ``latent_d``/``latent_y``/``latent_x``.  Supplying the wrong name
    should emit a warning so the user notices the search space is being
    ignored.
    """
    cfg_path = tmp_path / "search.json"
    cfg_path.write_text(json.dumps({
        # ``latent_dim`` is NOT a DIVA CLI attribute — runner must warn.
        "latent_dim": {"method": "suggest_categorical", "choices": [16]},
    }))

    args = _make_args(tmp_path, optuna_trials=1, optuna_config=str(cfg_path))

    def stub_train(a, outdir, *, verbose=True):
        return {"best_val": 1.0}

    with pytest.warns(UserWarning, match="latent_dim"):
        run_diva_optuna(args, stub_train)


def test_default_search_space_beta_max_covers_all_three_backbones():
    """beta_max range must cover diva-beta-vae's default 0.05 (β-VAE
    regime) AND the NB-likelihood DIVAs' 1.0.  A range that boxes out
    either side defeats the point of a shared search space.
    """
    spec = DEFAULT_DIVA_SEARCH_SPACE["beta_max"]
    assert spec["low"] <= 0.05
    assert spec["high"] >= 1.0
