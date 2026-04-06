"""Quantum distance estimation — k-NN anomaly detection using quantum distances."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from quantum_anomaly_detection.circuits.swap_test import (
    compute_distance_matrix,
    state_fidelity_distance,
)


def knn_anomaly_score(
    distance_matrix: np.ndarray, k: int = 5
) -> np.ndarray:
    """Compute k-NN anomaly score: average distance to k nearest neighbors.

    Higher score = more anomalous (farther from neighbors).
    """
    n = distance_matrix.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        # Sort distances, skip self (distance 0)
        dists = np.sort(distance_matrix[i])
        # Take k nearest (skip index 0 which is self with dist=0)
        k_nearest = dists[1 : k + 1]
        scores[i] = np.mean(k_nearest) if len(k_nearest) > 0 else 0.0
    return scores


def detect_anomalies_knn(
    X: np.ndarray,
    feature_map: QuantumCircuit,
    k: int = 5,
    contamination: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline: quantum distances -> k-NN scores -> threshold.

    Returns (anomaly_labels, scores).
    Labels: 1 = anomaly, 0 = normal.
    """
    D = compute_distance_matrix(X, feature_map)
    scores = knn_anomaly_score(D, k=k)
    threshold = np.percentile(scores, 100 * (1 - contamination))
    labels = (scores > threshold).astype(int)
    return labels, scores
