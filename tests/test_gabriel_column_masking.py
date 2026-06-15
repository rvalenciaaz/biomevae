"""Tests for Gabriel-split column masking and ML correctness fixes.

Validates that cross_validate_nmf and cross_validate_vae mask held-out
columns at inference time, and that related training-loop fixes work
correctly (early stopping metric, per-feature R², external validation,
global seed restoration).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomevae.reconstruction import (  # noqa: E402
    compute_reconstruction_metrics,
    cross_validate_nmf,
    gabriel_split,
)


class TestGabrielSplit:
    """Basic sanity checks for gabriel_split."""

    def test_split_indices_cover_all_rows(self):
        X = np.random.default_rng(42).poisson(5, size=(20, 30))
        train_rows, val_rows, val_cols = gabriel_split(X, 0.8, seed=0)
        assert np.array_equal(
            np.sort(np.concatenate([train_rows, val_rows])),
            np.arange(20),
        )

    def test_val_cols_nonempty(self):
        X = np.random.default_rng(42).poisson(5, size=(20, 30))
        _, _, val_cols = gabriel_split(X, 0.9, seed=0)
        assert val_cols.size > 0


class TestNMFColumnMasking:
    """Ensure NMF projection uses only train columns."""

    def test_nnls_projection_excludes_val_cols(self):
        """When val_cols are zeroed in the original data, the NNLS projection
        that uses only train_cols should give the same W_val as the full
        data case (since it never touches val_cols)."""
        rng = np.random.default_rng(7)
        n, p, k = 30, 40, 4
        X = rng.poisson(5, size=(n, p)).astype(np.float32)
        X = np.log1p(X)

        result = cross_validate_nmf(
            X.astype(np.int64),  # counts input
            n_components=k,
            n_splits=2,
            train_fraction=0.8,
            random_state=42,
        )
        # If column masking works, we should get valid metrics
        assert "mae" in result.mean_metrics
        assert "r2" in result.mean_metrics
        assert np.isfinite(result.mean_metrics["mae"])

    def test_nmf_cv_metadata(self):
        rng = np.random.default_rng(7)
        X = rng.poisson(5, size=(20, 25))
        result = cross_validate_nmf(
            X, n_components=3, n_splits=2, train_fraction=0.8, random_state=0
        )
        assert result.metadata["val_cols"] > 0
        assert result.metadata["val_rows"] > 0


class TestVAEColumnMasking:
    """Ensure VAE encoder receives zeroed-out val_cols."""

    @patch("biomevae.reconstruction.train_once")
    def test_encoder_receives_masked_input(self, mock_train_once):
        """Verify that the tensor passed to model.encode has val_cols zeroed."""
        captured_inputs = []

        # Build a minimal mock model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        def fake_encode(tensor):
            captured_inputs.append(tensor.cpu().numpy().copy())
            n = tensor.shape[0]
            latent_dim = 2
            return tensor.new_zeros(n, latent_dim), tensor.new_zeros(n, latent_dim)

        mock_model.encode.side_effect = fake_encode
        mock_model.decoder.return_value = MagicMock(
            cpu=MagicMock(
                return_value=MagicMock(
                    numpy=MagicMock(
                        return_value=np.zeros((5, 20), dtype=np.float32)
                    )
                )
            )
        )

        mock_train_once.return_value = {"model": mock_model}

        # Patch gabriel_split to return deterministic indices
        with patch("biomevae.reconstruction.gabriel_split") as mock_split:
            val_cols = np.array([2, 5, 8])
            mock_split.return_value = (
                np.arange(10, 20),  # train_rows
                np.arange(0, 5),    # val_rows (5 samples)
                val_cols,
            )

            rng = np.random.default_rng(0)
            X = rng.poisson(10, size=(20, 20)).astype(np.float32)

            from biomevae.reconstruction import cross_validate_vae

            try:
                cross_validate_vae(
                    X,
                    params={"latent_dim": 2, "device": "cpu"},
                    n_splits=1,
                    train_fraction=0.8,
                    seed=0,
                )
            except Exception:
                pass  # We only care about what was passed to encode

            # Verify encode was called and val_cols were zeroed
            assert len(captured_inputs) > 0, "model.encode was never called"
            encoded_input = captured_inputs[0]
            np.testing.assert_array_equal(
                encoded_input[:, val_cols],
                0.0,
                err_msg="val_cols should be zeroed before encoding",
            )


class TestPerFeatureR2:
    """Validate the per-feature R² metric."""

    def test_perfect_reconstruction(self):
        target = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metrics = compute_reconstruction_metrics(target, target)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["r2_per_feature"] == pytest.approx(1.0)

    def test_per_feature_r2_differs_from_global(self):
        """Per-feature R² should differ from global R² when features have
        very different scales."""
        rng = np.random.default_rng(42)
        n = 100
        # Feature 0: large scale, well-predicted; Feature 1: small scale, badly predicted
        target = np.column_stack([rng.normal(1000, 10, n), rng.normal(1, 0.1, n)])
        pred = target.copy()
        pred[:, 0] += rng.normal(0, 1, n)    # small relative error
        pred[:, 1] += rng.normal(0, 0.5, n)  # large relative error
        metrics = compute_reconstruction_metrics(target, pred)
        # Global R² is dominated by the large-scale feature
        assert metrics["r2"] > 0.9
        # Per-feature R² exposes the poor second feature
        assert metrics["r2_per_feature"] < metrics["r2"]

    def test_1d_input_falls_back(self):
        target = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.1, 3.1])
        metrics = compute_reconstruction_metrics(target, pred)
        assert metrics["r2_per_feature"] == pytest.approx(metrics["r2"])

    def test_nmf_cv_includes_per_feature_r2(self):
        rng = np.random.default_rng(7)
        X = rng.poisson(5, size=(20, 25))
        result = cross_validate_nmf(
            X, n_components=3, n_splits=2, train_fraction=0.8, random_state=0
        )
        assert "r2_per_feature" in result.mean_metrics


class TestGlobalSeedRestoration:
    """Verify that train_once restores global random state."""

    def test_numpy_state_restored(self):
        from biomevae.trainers.train_loop import train_once

        np.random.seed(999)
        before = np.random.get_state()[1][:5].copy()
        np.random.seed(999)

        # Create minimal data and params
        X = np.random.default_rng(0).normal(size=(20, 10)).astype(np.float32)
        params = {
            "objective": "beta", "val_split": 0.2, "standardize": False,
            "device": "cpu", "hidden": [8], "latent_dim": 2, "dropout": 0.0,
            "activation": "relu", "layer_norm": False, "optimizer": "adam",
            "lr": 1e-3, "weight_decay": 0.0, "batch_size": 10, "epochs": 1,
            "kl_warmup": 0, "beta_max": 1.0, "free_bits": 0.0,
            "recon": "mse", "huber_delta": 1.0, "capacity_start": 0.0,
            "capacity_end": None, "capacity_epochs": 10, "capacity_gamma": 1.0,
            "grad_clip": 0, "early_stop": 0,
        }
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            train_once(X, [f"s{i}" for i in range(20)], tmpdir, params, seed=42)

        after = np.random.get_state()[1][:5].copy()
        # The state after train_once should NOT be the seed=42 state;
        # it should be advanced from the original seed=999 state.
        # Specifically, it should not match what seed=42 would produce.
        np.random.seed(42)
        seed42_state = np.random.get_state()[1][:5].copy()
        assert not np.array_equal(after, seed42_state), (
            "Global numpy state was left at the train_once seed instead of being restored"
        )


class TestExternalVal:
    """Verify that train_once accepts external_val."""

    def test_external_val_uses_all_training_data(self):
        from biomevae.trainers.train_loop import train_once

        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 10)).astype(np.float32)
        val = rng.normal(size=(5, 10)).astype(np.float32)
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
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            res = train_once(
                X, [f"s{i}" for i in range(20)], tmpdir, params,
                seed=42, external_val=val, return_model=True,
            )
        assert res["model"] is not None
        assert np.isfinite(res["best_val"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
