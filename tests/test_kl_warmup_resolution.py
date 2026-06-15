"""Regression tests for the KL warmup resolver.

Design note — the resolver intentionally does NOT modify the schedule when
the user-supplied absolute ``kl_warmup`` exceeds ``epochs``. A long warmup
is a legitimate ML choice (β stays small for the whole run, keeping KL
pressure gentle while the decoder learns). An empirical sweep on the test
dataset showed that clamping long warmups down to ``epochs // 4`` induces
posterior collapse (KL → 0) for β_max ≥ 0.1 — a worse failure mode than
the original non-stationary-ELBO cosmetic issue. The resolver therefore
warns but leaves the schedule untouched. Only ``kl_warmup_frac`` (opt-in)
rewrites the warmup.

These tests lock in those exact semantics.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomevae.trainers.train_loop import _resolve_kl_warmup  # noqa: E402


class TestResolveKlWarmup:
    def test_fraction_is_run_relative(self):
        assert (
            _resolve_kl_warmup({"epochs": 100, "kl_warmup_frac": 0.25}, verbose=False)
            == 25
        )

    def test_fraction_rounds_to_nearest_epoch(self):
        assert (
            _resolve_kl_warmup({"epochs": 90, "kl_warmup_frac": 0.333}, verbose=False)
            == 30
        )

    def test_fraction_clamps_to_minimum_one(self):
        assert (
            _resolve_kl_warmup({"epochs": 5, "kl_warmup_frac": 0.01}, verbose=False)
            == 1
        )

    def test_fraction_overrides_absolute(self):
        # Fraction is the explicit opt-in — when both are set it wins.
        assert (
            _resolve_kl_warmup(
                {"epochs": 100, "kl_warmup": 300, "kl_warmup_frac": 0.2},
                verbose=False,
            )
            == 20
        )

    def test_absolute_below_epochs_is_preserved(self):
        assert (
            _resolve_kl_warmup({"epochs": 100, "kl_warmup": 40}, verbose=False)
            == 40
        )

    def test_absolute_equal_to_epochs_is_preserved(self):
        # β will saturate exactly at the last epoch; this is a valid
        # choice and is not silently modified.
        assert (
            _resolve_kl_warmup({"epochs": 100, "kl_warmup": 100}, verbose=False)
            == 100
        )

    def test_absolute_longer_than_epochs_is_preserved(self):
        # Historical default (300 warmup, 100 epochs): β reaches 1/3 of
        # β_max at the final epoch. This is the ML-effective slow-warmup
        # schedule; the resolver does NOT clamp it.
        assert (
            _resolve_kl_warmup({"epochs": 100, "kl_warmup": 300}, verbose=False)
            == 300
        )

    def test_absolute_overlong_warns(self, capsys):
        # It should emit a user-facing warning so over-long warmups are
        # at least visible in the training log.
        _resolve_kl_warmup({"epochs": 100, "kl_warmup": 300}, verbose=True)
        captured = capsys.readouterr()
        assert "kl_warmup" in captured.out.lower()
        assert "300" in captured.out and "100" in captured.out

    def test_zero_warmup_is_preserved(self):
        # Vanilla VAE case: no warmup, β stays at β_max from epoch 1.
        assert (
            _resolve_kl_warmup({"epochs": 100, "kl_warmup": 0}, verbose=False) == 0
        )

    def test_missing_kl_warmup_is_zero(self):
        assert _resolve_kl_warmup({"epochs": 100}, verbose=False) == 0

    def test_fraction_none_falls_back_to_absolute(self):
        # Optuna may pass kl_warmup_frac=None for vanilla; the resolver
        # must treat it as "not provided".
        assert (
            _resolve_kl_warmup(
                {"epochs": 200, "kl_warmup": 50, "kl_warmup_frac": None},
                verbose=False,
            )
            == 50
        )

    def test_none_absolute_value_is_zero(self):
        # argparse default None for --kl-warmup means "unset"; treat as 0.
        assert (
            _resolve_kl_warmup(
                {"epochs": 100, "kl_warmup": None, "kl_warmup_frac": None},
                verbose=False,
            )
            == 0
        )


class TestResolveInsideTrainLoop:
    """End-to-end: train_once must honour both parameter shapes."""

    def _base_params(self):
        return {
            "objective": "beta",
            "val_split": 0.2,
            "standardize": False,
            "device": "cpu",
            "hidden": [8],
            "latent_dim": 2,
            "dropout": 0.0,
            "activation": "relu",
            "layer_norm": False,
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 10,
            "epochs": 4,
            "beta_max": 1.0,
            "free_bits": 0.0,
            "recon": "mse",
            "huber_delta": 1.0,
            "capacity_start": 0.0,
            "capacity_end": None,
            "capacity_epochs": 10,
            "capacity_gamma": 1.0,
            "grad_clip": 0,
            "early_stop": 0,
        }

    def test_fractional_warmup_completes_training(self, tmp_path):
        from biomevae.trainers.train_loop import train_once

        X = np.random.default_rng(0).normal(size=(30, 8)).astype(np.float32)
        params = dict(self._base_params(), kl_warmup_frac=0.5)
        res = train_once(
            X, [f"s{i}" for i in range(30)], str(tmp_path), params, seed=1,
            verbose=False,
        )
        assert np.isfinite(res["best_val"])
        # 4 epochs * 0.5 = 2.
        assert res["config"]["kl_warmup"] == 2

    def test_overlong_absolute_warmup_is_preserved(self, tmp_path):
        """Regression: a slow warmup is ML-effective, NOT a bug.

        This locks in the decision that we do not silently clamp the
        schedule — training runs with the user-specified warmup even
        when it exceeds --epochs.
        """
        from biomevae.trainers.train_loop import train_once

        X = np.random.default_rng(0).normal(size=(30, 8)).astype(np.float32)
        params = dict(self._base_params(), kl_warmup=300)
        res = train_once(
            X, [f"s{i}" for i in range(30)], str(tmp_path), params, seed=1,
            verbose=False,
        )
        # The schedule is preserved verbatim.
        assert res["config"]["kl_warmup"] == 300


class TestTrainingLogSchema:
    """Every training loop must emit train_recon/val_recon so the unified
    single_study_figures plotter can show the stationary convergence
    signal on every model. PhILR-VAE, TreeNB-VAE and HGVAE-ZI use NB-NLL
    rather than MSE, so they alias train_recon/val_recon to their nll.
    """

    def test_shared_trainer_has_recon_columns(self, tmp_path):
        import pandas as pd
        from biomevae.trainers.train_loop import train_once

        X = np.random.default_rng(0).normal(size=(30, 8)).astype(np.float32)
        params = {
            "objective": "beta", "val_split": 0.2, "standardize": False,
            "device": "cpu", "hidden": [8], "latent_dim": 2, "dropout": 0.0,
            "activation": "relu", "layer_norm": False, "optimizer": "adam",
            "lr": 1e-3, "weight_decay": 0.0, "batch_size": 10, "epochs": 2,
            "kl_warmup": 0, "beta_max": 1.0, "free_bits": 0.0,
            "recon": "mse", "huber_delta": 1.0, "capacity_start": 0.0,
            "capacity_end": None, "capacity_epochs": 10, "capacity_gamma": 1.0,
            "grad_clip": 0, "early_stop": 0,
        }
        train_once(X, [f"s{i}" for i in range(30)], str(tmp_path), params,
                   seed=1, verbose=False)
        df = pd.read_csv(tmp_path / "training_log.tsv", sep="\t")
        assert "train_recon" in df.columns
        assert "val_recon" in df.columns

    def test_philrvae_log_has_recon_aliases(self):
        """PhILR-VAE: train_recon/val_recon must alias the reconstruction NLL
        columns so the unified plotter picks the curve up."""
        import re
        src = Path(__file__).resolve().parents[1] / "src" / "biomevae" / "cli" / "vae_train_philrvae.py"
        text = src.read_text()
        assert re.search(r'"train_recon"\s*:\s*t_recon\s*/\s*n_t', text), (
            "train_recon alias missing in vae_train_philrvae log row"
        )
        assert re.search(r'"val_recon"\s*:\s*v_recon\s*/\s*n_v', text), (
            "val_recon alias missing in vae_train_philrvae log row"
        )

    def test_tree_dtm_vae_log_has_recon_aliases(self):
        """TreeDTM-VAE uses DTM/Dirichlet NLL; train_recon/val_recon must alias."""
        import re
        src = Path(__file__).resolve().parents[1] / "src" / "biomevae" / "cli" / "vae_train_tree_dtm_vae.py"
        text = src.read_text()
        assert re.search(r'"train_recon"\s*:\s*t_recon\s*/\s*n', text), (
            "train_recon alias missing in vae_train_tree_dtm_vae training row"
        )
        assert re.search(r'row\["val_recon"\]\s*=\s*val_nll', text), (
            "val_recon alias missing in vae_train_tree_dtm_vae validation row"
        )

    def test_hgvae_zi_log_has_recon(self):
        """HGVAE-ZI already computed recon — make sure it still does."""
        import re
        src = Path(__file__).resolve().parents[1] / "src" / "biomevae" / "cli" / "vae_train_hgvae_zi.py"
        text = src.read_text()
        assert '"train_recon"' in text
        assert 'row["val_recon"]' in text


class TestPlotTrainingCurvesRecon:
    """``biomevae-plot-training-curves`` must produce recon curves for
    PhILR-VAE alongside MSE-based models.

    Before the fix the linear y-axis collapsed the small-scale MSE
    curves and/or the large-scale NB-NLL curves so that one of the two
    families disappeared from the plot. The fix is a log y-axis and
    graceful handling of missing recon columns (older logs). These
    tests lock in both behaviours.
    """

    def _write_log(self, path, **columns):
        import pandas as pd
        pd.DataFrame(columns).to_csv(path, sep="\t", index=False)

    def test_recon_plot_shows_philrvae_with_mse_model(self, tmp_path):
        """Both PhILR-VAE (NB-NLL ~100s) and base (MSE ~0.1) must be plotted."""
        from biomevae.cli.plot_training_curves import main

        philrvae_log = tmp_path / "philrvae.tsv"
        base_log = tmp_path / "base.tsv"
        self._write_log(
            philrvae_log,
            epoch=[1, 2, 3],
            train_loss=[400.0, 300.0, 250.0],
            val_loss=[420.0, 320.0, 260.0],
            train_recon=[398.0, 298.0, 248.0],
            val_recon=[418.0, 318.0, 258.0],
        )
        self._write_log(
            base_log,
            epoch=[1, 2, 3],
            train_loss=[0.5, 0.3, 0.2],
            val_loss=[0.6, 0.35, 0.25],
            train_recon=[0.45, 0.25, 0.15],
            val_recon=[0.55, 0.30, 0.20],
        )
        outdir = tmp_path / "curves"
        main([
            "--log", f"philrvae={philrvae_log}",
            "--log", f"base={base_log}",
            "--metric", "recon",
            "--output", str(outdir),
        ])
        assert (outdir / "training_recon_curves.png").exists()
        assert (outdir / "validation_recon_curves.png").exists()
        assert (outdir / "train_val_recon_curves.png").exists()

    def test_recon_plot_skips_logs_missing_columns(self, tmp_path, capsys):
        """Older logs without train_recon/val_recon should be skipped with a warning."""
        from biomevae.cli.plot_training_curves import main

        philrvae_log = tmp_path / "philrvae.tsv"
        old_log = tmp_path / "old.tsv"
        self._write_log(
            philrvae_log,
            epoch=[1, 2, 3],
            train_loss=[400.0, 300.0, 250.0],
            val_loss=[420.0, 320.0, 260.0],
            train_recon=[398.0, 298.0, 248.0],
            val_recon=[418.0, 318.0, 258.0],
        )
        self._write_log(
            old_log,
            epoch=[1, 2, 3],
            train_loss=[0.5, 0.3, 0.2],
            val_loss=[0.6, 0.35, 0.25],
        )
        outdir = tmp_path / "curves"
        main([
            "--log", f"philrvae={philrvae_log}",
            "--log", f"old={old_log}",
            "--metric", "recon",
            "--output", str(outdir),
        ])
        captured = capsys.readouterr()
        assert "old" in captured.out
        assert "missing column" in captured.out
        # PhILR-VAE panel must still be written even though the old
        # log was skipped.
        assert (outdir / "train_val_recon_curves.png").exists()

    def test_default_yscale_is_log(self):
        """Log scale is the default so NB and MSE curves coexist."""
        from biomevae.cli.plot_training_curves import parse_args

        args = parse_args(["--log", "foo=/tmp/foo.tsv"])
        assert args.yscale == "log"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
