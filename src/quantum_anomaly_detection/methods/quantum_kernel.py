"""Quantum kernel method — compute kernel matrix and use with One-Class SVM."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from sklearn.svm import OneClassSVM

from quantum_anomaly_detection.circuits.feature_maps import assign_features


def compute_kernel_entry(
    x1: np.ndarray, x2: np.ndarray, feature_map: QuantumCircuit
) -> float:
    """Compute kernel K(x1, x2) = |<phi(x1)|phi(x2)>|^2."""
    circ1 = assign_features(feature_map, x1)
    circ2 = assign_features(feature_map, x2)
    sv1 = Statevector(circ1)
    sv2 = Statevector(circ2)
    return float(state_fidelity(sv1, sv2))


def compute_kernel_matrix(
    X: np.ndarray,
    feature_map: QuantumCircuit,
    Y: np.ndarray | None = None,
) -> np.ndarray:
    """Compute kernel matrix.

    If Y is None, compute symmetric K(X, X).
    If Y is given, compute K(X, Y) (rectangular).
    """
    if Y is None:
        n = len(X)
        K = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                k = compute_kernel_entry(X[i], X[j], feature_map)
                K[i, j] = k
                K[j, i] = k
        return K
    else:
        K = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                K[i, j] = compute_kernel_entry(X[i], Y[j], feature_map)
        return K


def quantum_kernel_svm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_map: QuantumCircuit,
    nu: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, OneClassSVM]:
    """Full pipeline: quantum kernel matrix + One-Class SVM.

    Returns (predictions, scores, fitted_model).
    Predictions: +1 = normal, -1 = anomaly.
    """
    K_train = compute_kernel_matrix(X_train, feature_map)
    K_test = compute_kernel_matrix(X_test, feature_map, Y=X_train)

    model = OneClassSVM(kernel="precomputed", nu=nu)
    model.fit(K_train)

    predictions = model.predict(K_test)
    scores = model.decision_function(K_test)

    return predictions, scores, model
