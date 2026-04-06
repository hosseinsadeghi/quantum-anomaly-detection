"""Graph/network data loading and preprocessing — KDD Cup 99 network intrusion."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


def load_kdd_cup(
    subsample: int = 5000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Load KDD Cup 99 network intrusion dataset.

    Returns (X, y_binary) where y=1 is attack (anomaly).
    Categorical features are label-encoded.
    """
    data = fetch_kddcup99(subset="SA", as_frame=False, percent10=True)
    X_raw = data.data
    y_raw = data.target

    # Encode categorical columns (indices 1, 2, 3 are categorical)
    X = np.copy(X_raw).astype(object)
    for col_idx in [1, 2, 3]:
        le = LabelEncoder()
        X[:, col_idx] = le.fit_transform(X[:, col_idx].astype(str))
    X = X.astype(np.float64)

    # Binary labels: normal vs attack
    y = np.array([0 if label == b"normal." else 1 for label in y_raw])

    # Subsample
    rng = np.random.default_rng(seed)
    n = min(subsample, len(X))
    idx = rng.choice(len(X), size=n, replace=False)

    return X[idx], y[idx]


def preprocess_graph_features(
    X: np.ndarray,
    n_components: int = 8,
    fit_data: np.ndarray | None = None,
) -> np.ndarray:
    """StandardScaler + PCA, then scale to [0, pi]."""
    source = fit_data if fit_data is not None else X

    scaler = StandardScaler()
    scaler.fit(source)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(scaler.transform(source))
    X_pca = pca.transform(X_scaled)

    minmax = MinMaxScaler(feature_range=(0, np.pi))
    minmax.fit(pca.transform(scaler.transform(source)))
    return minmax.transform(X_pca)


def build_adjacency_from_features(
    X: np.ndarray, k: int = 5
) -> np.ndarray:
    """Build k-NN adjacency matrix from feature vectors.

    Returns a symmetric binary adjacency matrix.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    n = len(X)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in indices[i, 1:]:  # Skip self
            adj[i, j] = 1
            adj[j, i] = 1

    return adj
