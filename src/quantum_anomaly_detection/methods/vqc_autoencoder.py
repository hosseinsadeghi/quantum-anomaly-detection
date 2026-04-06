"""Variational Quantum Circuit Autoencoder — training and anomaly detection."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit

from quantum_anomaly_detection.circuits.autoencoder import (
    build_autoencoder_circuit,
    batch_reconstruction_loss,
    reconstruction_loss,
)


def train_autoencoder(
    X_train: np.ndarray,
    n_latent: int,
    encoder_reps: int = 2,
    decoder_reps: int = 2,
    maxiter: int = 200,
    seed: int = 42,
    method: str = "COBYLA",
) -> tuple[np.ndarray, QuantumCircuit, list[float]]:
    """Train a VQC autoencoder on normal data.

    Returns (optimal_params, circuit, cost_history).
    """
    n_qubits = X_train.shape[1]
    circuit = build_autoencoder_circuit(n_qubits, n_latent, encoder_reps, decoder_reps)

    # Count variational parameters (exclude data 'x_*' params)
    var_params = [p for p in circuit.parameters if not p.name.startswith("x")]
    n_params = len(var_params)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=n_params)

    cost_history = []

    def cost_fn(params):
        loss = batch_reconstruction_loss(params, circuit, X_train, n_latent)
        cost_history.append(loss)
        return loss

    result = minimize(cost_fn, x0, method=method, options={"maxiter": maxiter})

    return result.x, circuit, cost_history


def score_anomalies(
    X: np.ndarray,
    params: np.ndarray,
    circuit: QuantumCircuit,
    n_latent: int,
) -> np.ndarray:
    """Compute reconstruction error for each sample. Higher = more anomalous."""
    scores = np.array([
        reconstruction_loss(params, circuit, x, n_latent) for x in X
    ])
    return scores


def detect_anomalies(
    scores: np.ndarray,
    threshold: float | None = None,
    contamination: float = 0.05,
) -> np.ndarray:
    """Threshold anomaly scores to produce binary labels.

    If threshold is None, use percentile based on contamination.
    Returns 1 = anomaly, 0 = normal.
    """
    if threshold is None:
        threshold = np.percentile(scores, 100 * (1 - contamination))
    return (scores > threshold).astype(int)
