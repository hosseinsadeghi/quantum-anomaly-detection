"""QAOA-based clustering for anomaly detection.

Formulates 2-cluster partitioning as QUBO, solves with QAOA,
then labels small/distant clusters as anomalies.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from quantum_anomaly_detection.circuits.qaoa import (
    build_clustering_hamiltonian,
    build_qaoa_circuit,
    evaluate_qaoa_cost,
    decode_qaoa_solution,
)


def run_qaoa_clustering(
    X: np.ndarray,
    reps: int = 2,
    maxiter: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Run QAOA clustering on a small dataset.

    Args:
        X: Data points, shape (n_points, n_features). n_points should be <= 20.
        reps: Number of QAOA layers.
        maxiter: Max optimizer iterations.
        seed: Random seed.

    Returns (cluster_labels, cost_history).
    Labels are 0 or 1 for each point.
    """
    distance_matrix = squareform(pdist(X, metric="euclidean"))
    # Normalize distances for numerical stability
    if distance_matrix.max() > 0:
        distance_matrix = distance_matrix / distance_matrix.max()

    cost_op = build_clustering_hamiltonian(distance_matrix)
    circuit = build_qaoa_circuit(cost_op, reps=reps)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)

    cost_history = []

    def cost_fn(params):
        val = evaluate_qaoa_cost(params, circuit, cost_op)
        cost_history.append(val)
        return val

    result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": maxiter})

    labels = decode_qaoa_solution(result.x, circuit)
    return labels, cost_history


def identify_anomaly_clusters(
    labels: np.ndarray,
    X: np.ndarray,
    min_cluster_fraction: float = 0.1,
) -> np.ndarray:
    """Label points in small clusters as anomalies.

    A cluster is "small" if it has fewer than min_cluster_fraction of total points.
    Returns binary anomaly labels (1 = anomaly).
    """
    unique_labels = np.unique(labels)
    n = len(labels)
    anomaly = np.zeros(n, dtype=int)

    for lbl in unique_labels:
        mask = labels == lbl
        if mask.sum() < n * min_cluster_fraction:
            anomaly[mask] = 1

    return anomaly
