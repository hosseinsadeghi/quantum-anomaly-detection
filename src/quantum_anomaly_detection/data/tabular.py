"""Tabular data loading and preprocessing — credit card fraud and synthetic blobs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_blobs
from sklearn.model_selection import train_test_split


def load_creditcard(
    subsample: int = 5000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Download credit card fraud dataset from OpenML (ID 1597).

    Returns (X, y) where y=1 is anomaly (fraud).
    Subsamples to keep runtime manageable while preserving class ratio.
    """
    data = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    df = data.frame
    X = df.drop(columns=["Class"]).values.astype(np.float64)
    y = df["Class"].astype(int).values

    rng = np.random.default_rng(seed)
    n = min(subsample, len(X))
    # Stratified subsample
    idx_normal = np.where(y == 0)[0]
    idx_anomaly = np.where(y == 1)[0]
    frac = len(idx_anomaly) / len(y)
    n_anomaly = max(1, int(n * frac))
    n_normal = n - n_anomaly

    chosen_normal = rng.choice(idx_normal, size=n_normal, replace=False)
    chosen_anomaly = rng.choice(
        idx_anomaly, size=min(n_anomaly, len(idx_anomaly)), replace=False
    )
    idx = np.concatenate([chosen_normal, chosen_anomaly])
    rng.shuffle(idx)

    return X[idx], y[idx]


def load_synthetic_blobs(
    n_samples: int = 1000,
    n_features: int = 4,
    anomaly_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian blobs with uniform outliers as anomalies."""
    rng = np.random.default_rng(seed)
    n_normal = int(n_samples * (1 - anomaly_fraction))
    n_anomaly = n_samples - n_normal

    X_normal, _ = make_blobs(
        n_samples=n_normal,
        n_features=n_features,
        centers=3,
        cluster_std=1.0,
        random_state=seed,
    )

    # Anomalies: uniform in a wider range
    lo, hi = X_normal.min() - 3, X_normal.max() + 3
    X_anomaly = rng.uniform(lo, hi, size=(n_anomaly, n_features))

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def preprocess_tabular(
    X: np.ndarray,
    n_components: int | None = None,
    fit_data: np.ndarray | None = None,
) -> np.ndarray:
    """StandardScaler + optional PCA, then scale to [0, pi].

    If fit_data is provided, the scaler/PCA are fit on fit_data and applied to X.
    Otherwise X is used for fitting.
    """
    from quantum_anomaly_detection.data.preprocessing import scale_to_quantum_range
    return scale_to_quantum_range(X, n_components=n_components, fit_data=fit_data)
