"""Swap test circuit and fidelity-based quantum distance estimation."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from quantum_anomaly_detection.circuits.feature_maps import assign_features


def build_swap_test_circuit(n_qubits: int) -> QuantumCircuit:
    """Build a swap test circuit for two n-qubit registers.

    Layout: 1 ancilla + n_qubits (register A) + n_qubits (register B)
    Total qubits: 2 * n_qubits + 1

    The ancilla is measured. P(ancilla=0) = (1 + |<A|B>|^2) / 2
    So fidelity = 2 * P(0) - 1.
    """
    total = 2 * n_qubits + 1
    qc = QuantumCircuit(total, 1, name="SwapTest")

    ancilla = 0
    reg_a = list(range(1, n_qubits + 1))
    reg_b = list(range(n_qubits + 1, total))

    # Hadamard on ancilla
    qc.h(ancilla)

    # Controlled swaps between register A and B
    for i in range(n_qubits):
        qc.cswap(ancilla, reg_a[i], reg_b[i])

    # Hadamard on ancilla
    qc.h(ancilla)

    # Measure ancilla
    qc.measure(ancilla, 0)

    return qc


def state_fidelity_distance(
    x1: np.ndarray,
    x2: np.ndarray,
    feature_map: QuantumCircuit,
) -> float:
    """Compute quantum distance between two data points.

    Distance = sqrt(1 - |<phi(x1)|phi(x2)>|^2)

    Uses exact statevector computation (no sampling noise).
    """
    circ1 = assign_features(feature_map, x1)
    circ2 = assign_features(feature_map, x2)

    sv1 = Statevector(circ1)
    sv2 = Statevector(circ2)

    fid = state_fidelity(sv1, sv2)
    return float(np.sqrt(max(0.0, 1.0 - fid)))


def compute_distance_matrix(
    X: np.ndarray,
    feature_map: QuantumCircuit,
) -> np.ndarray:
    """Compute full NxN distance matrix using state fidelity.

    Exploits symmetry: only computes upper triangle.
    """
    n = len(X)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = state_fidelity_distance(X[i], X[j], feature_map)
            D[i, j] = d
            D[j, i] = d

    return D
