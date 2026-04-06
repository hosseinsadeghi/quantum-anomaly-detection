"""QAOA-based anomaly detection via regression residuals.

Fits a classical regression model, computes residuals, then uses QAOA
to solve the binary thresholding problem (which points are anomalies?).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge

from quantum_anomaly_detection.circuits.qaoa import (
    build_thresholding_hamiltonian,
    build_qaoa_circuit,
    evaluate_qaoa_cost,
    decode_qaoa_solution,
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

    # If y is multi-dimensional, take mean residual per sample
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
    # Normalize residuals
    if residuals.max() > 0:
        residuals_norm = residuals / residuals.max()
    else:
        residuals_norm = residuals

    cost_op = build_thresholding_hamiltonian(residuals_norm, penalty=penalty)
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
