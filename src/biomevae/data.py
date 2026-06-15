from typing import List, Tuple, Optional
import os
import warnings
import numpy as np
import pandas as pd

__all__ = [
    "load_matrix",
    "train_val_split",
    "train_val_split_groups",
    "standardize_train_only",
    "save_scaler",
]

def load_matrix(path: str, log1p: bool) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if df.shape[1] < 3:
        raise SystemExit("Expected at least 3 columns: clade_name, NCBI_tax_id, and >=1 sample columns.")
    sample_cols = df.columns[2:]
    numeric = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    n_invalid = int(numeric.isna().sum().sum())
    if n_invalid > 0:
        warnings.warn(
            f"load_matrix: coerced {n_invalid} non-numeric/missing values to 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
    X_feat_sample = numeric.fillna(0.0).to_numpy()
    X = X_feat_sample.T.astype(np.float32)  # [samples, features]
    if log1p:
        X = np.log1p(X).astype(np.float32)
    return X, list(sample_cols)

def train_val_split(n: int, val_frac: float, seed: int):
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in the open interval (0, 1).")
    if n < 2:
        raise ValueError("At least two samples are required to create train/val splits.")

    idx = np.arange(n, dtype=int)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    val_size = max(1, int(round(n * val_frac)))
    val_size = min(val_size, n - 1)  # ensure at least one training sample remains
    return idx[val_size:], idx[:val_size]  # train_idx, val_idx


def train_val_split_groups(n: int, val_frac: float, seed: int, groups: np.ndarray):
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in the open interval (0, 1).")
    if n < 2:
        raise ValueError("At least two samples are required to create train/val splits.")
    if len(groups) != n:
        raise ValueError("groups must have the same length as the dataset.")
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Group split requires at least two unique groups.")

    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(n), groups=groups))
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Group split produced an empty train or validation set.")
    return train_idx, val_idx

def standardize_train_only(X: np.ndarray, train_idx: np.ndarray):
    mean = X[train_idx].mean(axis=0, keepdims=True)
    std = X[train_idx].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    scaler = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return Xs.astype(np.float32), scaler

def save_scaler(scaler: Optional[dict], outdir: str):
    if scaler is not None:
        os.makedirs(outdir, exist_ok=True)
        np.savez_compressed(os.path.join(outdir, "feature_scaler.npz"),
                            mean=scaler["mean"], std=scaler["std"])
