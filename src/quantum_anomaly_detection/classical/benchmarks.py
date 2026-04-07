"""Classical anomaly detection benchmarks and simulated annealing."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope


def run_isolation_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    contamination: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Isolation Forest. Returns (predictions, scores).

    Predictions: +1 = normal, -1 = anomaly.
    """
    model = IsolationForest(contamination=contamination, random_state=seed)
    model.fit(X_train)
    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)
    return predictions, scores


def run_svm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    kernel: str = "rbf",
    nu: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """One-Class SVM. Returns (predictions, scores)."""
    model = OneClassSVM(kernel=kernel, nu=nu)
    model.fit(X_train)
    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)
    return predictions, scores


def run_lof(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Local Outlier Factor. Returns (predictions, scores)."""
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, novelty=True
    )
    model.fit(X_train)
    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)
    return predictions, scores


def run_dbscan(
    X: np.ndarray, eps: float = 0.5, min_samples: int = 5
) -> np.ndarray:
    """DBSCAN clustering. Returns labels where -1 = anomaly."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)


def run_elliptic_envelope(
    X_train: np.ndarray,
    X_test: np.ndarray,
    contamination: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Elliptic Envelope (Mahalanobis distance). Returns (predictions, scores)."""
    model = EllipticEnvelope(contamination=contamination)
    model.fit(X_train)
    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)
    return predictions, scores


def run_simulated_annealing(
    cost_matrix: np.ndarray,
    n_iter: int = 10000,
    temp_init: float = 10.0,
    cooling_rate: float = 0.995,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Simulated annealing for 2-cluster MaxCut (same objective as QAOA).

    Minimizes: sum of D[i,j] for pairs in the SAME cluster (MaxCut =
    maximize cut weight = minimize same-cluster weight), plus a gentle
    balance penalty.

    Returns (cluster_labels, cost_history).
    """
    rng = np.random.default_rng(seed)
    n = cost_matrix.shape[0]

    state = rng.integers(0, 2, size=n)

    nonzero = cost_matrix[cost_matrix > 0]
    mean_d = float(nonzero.mean()) if len(nonzero) > 0 else 1.0
    balance_weight = mean_d / (2 * n)

    def compute_cost(s):
        cost = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    cost += cost_matrix[i, j]
        n1 = int(s.sum())
        n0 = n - n1
        cost += balance_weight * (n0 - n1) ** 2
        return cost

    current_cost = compute_cost(state)
    best_state = state.copy()
    best_cost = current_cost
    cost_history = [current_cost]

    temp = temp_init
    for _ in range(n_iter):
        # Flip one random bit
        flip_idx = rng.integers(0, n)
        new_state = state.copy()
        new_state[flip_idx] = 1 - new_state[flip_idx]

        new_cost = compute_cost(new_state)
        delta = new_cost - current_cost

        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            state = new_state
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = state.copy()

        temp *= cooling_rate
        cost_history.append(current_cost)

    return best_state, cost_history
