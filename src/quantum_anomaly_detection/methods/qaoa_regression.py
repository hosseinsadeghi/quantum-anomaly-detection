"""QAOA-based anomaly detection via regression residuals.

Fits a classical regression model, computes residuals, then uses QAOA
to solve the binary thresholding problem (which points are anomalies?).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from quantum_anomaly_detection.circuits.qaoa import (
    build_thresholding_hamiltonian,
    optimize_qaoa,
)


def fit_regression(
    X: np.ndarray, y: np.ndarray, model_type: str = "linear"
) -> tuple[object, np.ndarray]:
    """Fit a classical regression and return (model, residuals).

    For time series: X = previous windows, y = next window values.
    """
    if model_type == "ridge":
        model = Ridge(alpha=1.0)
    else:
        model = LinearRegression()

    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = np.abs(y - y_pred)

    if residuals.ndim > 1:
        residuals = residuals.mean(axis=1)

    return model, residuals


def run_qaoa_thresholding(
    residuals: np.ndarray,
    penalty: float = 0.5,
    reps: int = 2,
    maxiter: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Use QAOA to solve the anomaly/normal binary classification.

    Args:
        residuals: Absolute regression residuals, one per data point.
                   Length should be <= 16 for tractable QAOA.
        penalty: Smoothness penalty weight.
        reps: Number of QAOA layers.
        maxiter: Max optimizer iterations.
        seed: Random seed.

    Returns (anomaly_labels, cost_history).
    """
    if residuals.max() > 0:
        residuals_norm = residuals / residuals.max()
    else:
        residuals_norm = residuals

    cost_op = build_thresholding_hamiltonian(residuals_norm, penalty=penalty)
    return optimize_qaoa(cost_op, reps=reps, maxiter=maxiter, seed=seed)
