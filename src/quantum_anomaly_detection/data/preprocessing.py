"""Shared preprocessing pipeline for quantum feature encoding.

All data types follow the same final steps: StandardScaler -> PCA -> scale to [0, pi].
This module provides the shared implementation.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


def scale_to_quantum_range(
    X: np.ndarray,
    n_components: int | None = None,
    fit_data: np.ndarray | None = None,
) -> np.ndarray:
    """StandardScaler + optional PCA + scale to [0, pi].

    Args:
        X: Data to transform.
        n_components: If set, reduce dimensionality via PCA.
        fit_data: If provided, fit scaler/PCA on this data instead of X
                  (use training data here to avoid test leakage).

    Returns:
        Transformed data in [0, pi] range, suitable for quantum feature maps.
    """
    source = fit_data if fit_data is not None else X

    scaler = StandardScaler()
    scaler.fit(source)
    X_scaled = scaler.transform(X)
    source_scaled = scaler.transform(source)

    if n_components is not None and n_components < X.shape[1]:
        pca = PCA(n_components=n_components)
        pca.fit(source_scaled)
        X_scaled = pca.transform(X_scaled)
        source_scaled = pca.transform(source_scaled)

    minmax = MinMaxScaler(feature_range=(0, np.pi))
    minmax.fit(source_scaled)
    return minmax.transform(X_scaled)
