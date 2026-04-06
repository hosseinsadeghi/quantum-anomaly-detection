"""QAOA circuits — cost Hamiltonians and circuit construction.

Supports two QUBO formulations:
1. Clustering: partition n points into 2 clusters minimizing intra-cluster distance
2. Thresholding: binary classification of residuals as anomaly/normal
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator


def build_clustering_hamiltonian(distance_matrix: np.ndarray) -> SparsePauliOp:
    """Build QUBO Hamiltonian for 2-cluster partitioning.

    For n points with distance matrix D, we assign qubit i to cluster 0 (|0>)
    or cluster 1 (|1>). Cost minimizes intra-cluster distances:

        H = sum_{i<j} D[i,j]/2 * (I - Z_i Z_j)

    Points in the same cluster (same Z eigenvalue) contribute 0;
    points in different clusters contribute D[i,j].
    We minimize this to keep close points together.
    """
    n = distance_matrix.shape[0]
    terms = []

    for i in range(n):
        for j in range(i + 1, n):
            d = distance_matrix[i, j]
            if abs(d) < 1e-10:
                continue

            # D[i,j]/2 * I term (constant shift, can include for completeness)
            label_i = "I" * n
            terms.append((label_i, d / 2))

            # -D[i,j]/2 * Z_i Z_j term
            label_zz = list("I" * n)
            # SparsePauliOp uses little-endian: index 0 is rightmost
            label_zz[n - 1 - i] = "Z"
            label_zz[n - 1 - j] = "Z"
            terms.append(("".join(label_zz), -d / 2))

    return SparsePauliOp.from_list(terms).simplify()


def build_thresholding_hamiltonian(
    residuals: np.ndarray,
    penalty: float = 1.0,
) -> SparsePauliOp:
    """Build QUBO Hamiltonian for anomaly thresholding.

    Qubit i = |1> means point i is labeled anomaly.
    Cost encourages labeling points with large |residual| as anomalies:

        H = -sum_i |r_i| * Z_i + penalty * sum_{i<j} (1 - Z_i Z_j) / 4

    The first term rewards labeling high-residual points as anomalies.
    The penalty term encourages spatial smoothness (nearby points get same label).
    """
    n = len(residuals)
    terms = []

    for i in range(n):
        label = list("I" * n)
        label[n - 1 - i] = "Z"
        # Negative sign: minimize H means maximize Z_i for large |r_i|
        # Z_i = +1 for |0> (normal), Z_i = -1 for |1> (anomaly)
        # We want anomaly when |r_i| is large, so we use +|r_i| * Z_i
        # to penalize normal label for large residuals
        terms.append(("".join(label), float(np.abs(residuals[i]))))

    # Smoothness penalty
    if penalty > 0:
        for i in range(n):
            for j in range(i + 1, n):
                label_zz = list("I" * n)
                label_zz[n - 1 - i] = "Z"
                label_zz[n - 1 - j] = "Z"
                terms.append(("".join(label_zz), -penalty / 4))
                terms.append(("I" * n, penalty / 4))

    return SparsePauliOp.from_list(terms).simplify()


def build_qaoa_circuit(
    cost_op: SparsePauliOp,
    reps: int = 2,
) -> QuantumCircuit:
    """Build QAOA circuit from a cost operator using Qiskit's qaoa_ansatz."""
    return qaoa_ansatz(cost_op, reps=reps)


def evaluate_qaoa_cost(
    params: np.ndarray,
    circuit: QuantumCircuit,
    cost_op: SparsePauliOp,
) -> float:
    """Evaluate <psi(params)|cost_op|psi(params)> using StatevectorEstimator."""
    bound = circuit.assign_parameters(dict(zip(circuit.parameters, params)))
    sv = Statevector(bound)
    expectation = sv.expectation_value(cost_op)
    return float(np.real(expectation))


def decode_qaoa_solution(
    params: np.ndarray,
    circuit: QuantumCircuit,
) -> np.ndarray:
    """Get the most probable bitstring from the optimized QAOA state."""
    bound = circuit.assign_parameters(dict(zip(circuit.parameters, params)))
    sv = Statevector(bound)
    probs = sv.probabilities_dict()
    best = max(probs, key=probs.get)
    # Convert bitstring to array (big-endian: leftmost bit = qubit n-1)
    return np.array([int(b) for b in best])
