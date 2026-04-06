"""Image data loading and preprocessing — MNIST anomaly detection setup."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


def load_mnist_anomaly(
    normal_digit: int = 0,
    anomaly_digits: tuple[int, ...] = (1,),
    n_normal: int = 500,
    n_anomaly: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST and set up as anomaly detection problem.

    Normal class = normal_digit, anomalies = anomaly_digits.
    Returns (images_flat, labels) where labels=1 is anomaly.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_all = mnist.data.astype(np.float64)
    y_all = mnist.target.astype(int)

    rng = np.random.default_rng(seed)

    # Normal samples
    idx_normal = np.where(y_all == normal_digit)[0]
    chosen_normal = rng.choice(idx_normal, size=min(n_normal, len(idx_normal)), replace=False)

    # Anomaly samples
    idx_anomaly = np.where(np.isin(y_all, anomaly_digits))[0]
    chosen_anomaly = rng.choice(idx_anomaly, size=min(n_anomaly, len(idx_anomaly)), replace=False)

    X = np.vstack([X_all[chosen_normal], X_all[chosen_anomaly]])
    y = np.concatenate([np.zeros(len(chosen_normal)), np.ones(len(chosen_anomaly))])

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def preprocess_images(
    X: np.ndarray,
    n_components: int = 8,
    fit_data: np.ndarray | None = None,
) -> np.ndarray:
    """PCA reduction from 784 dims to n_components, then scale to [0, pi]."""
    source = fit_data if fit_data is not None else X

    # Normalize pixel values to [0, 1] first
    X_norm = X / 255.0
    source_norm = source / 255.0

    scaler = StandardScaler()
    scaler.fit(source_norm)
    X_scaled = scaler.transform(X_norm)

    pca = PCA(n_components=n_components)
    pca.fit(scaler.transform(source_norm))
    X_pca = pca.transform(X_scaled)

    minmax = MinMaxScaler(feature_range=(0, np.pi))
    minmax.fit(pca.transform(scaler.transform(source_norm)))
    return minmax.transform(X_pca)
