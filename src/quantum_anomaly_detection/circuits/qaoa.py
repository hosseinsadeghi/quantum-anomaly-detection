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


def build_clustering_hamiltonian(
    distance_matrix: np.ndarray,
    balance_weight: float | None = None,
) -> SparsePauliOp:
    """Build MaxCut-style QUBO Hamiltonian for balanced 2-cluster partitioning.

    For n points with distance matrix D, qubit i in |0> means cluster 0,
    |1> means cluster 1. The Hamiltonian has two terms:

    1. MaxCut cost: sum_{i<j} +D[i,j]/2 * Z_i Z_j
       Minimized when Z_i Z_j = -1 (different clusters) for large D[i,j].
       This places distant points in different clusters.

    2. Balance penalty: alpha * (sum_i Z_i)^2
       Gentle penalty for uneven cluster sizes. Expands to Z_i Z_j terms.
       Default alpha = mean(D) / (2*n) — weak enough not to override MaxCut.
    """
    n = distance_matrix.shape[0]
    if balance_weight is None:
        nonzero = distance_matrix[distance_matrix > 0]
        mean_d = float(nonzero.mean()) if len(nonzero) > 0 else 1.0
        balance_weight = mean_d / (2 * n)

    terms = []

    for i in range(n):
        for j in range(i + 1, n):
            d = distance_matrix[i, j]
            if abs(d) < 1e-10:
                continue
            label_zz = list("I" * n)
            label_zz[n - 1 - i] = "Z"
            label_zz[n - 1 - j] = "Z"
            # MaxCut: +d/2 (positive!), balance: +alpha
            terms.append(("".join(label_zz), d / 2 + balance_weight))

    if not terms:
        terms.append(("I" * n, 0.0))

    return SparsePauliOp.from_list(terms).simplify()


def build_thresholding_hamiltonian(
    residuals: np.ndarray,
    penalty: float = 1.0,
) -> SparsePauliOp:
    """Build QUBO Hamiltonian for anomaly thresholding.

    Qubit i in |0> = normal, |1> = anomaly. Z_i eigenvalue is +1 for |0>, -1 for |1>.

    H = +sum_i |r_i| * Z_i + penalty * sum_{i<j} (1 - Z_i Z_j) / 4

    First term: +|r_i| * Z_i means minimizing H prefers Z_i = -1 (anomaly)
    when |r_i| is large. Points with small residuals stay normal (Z_i = +1).

    Second term: smoothness penalty encouraging nearby points to share labels.
    """
    n = len(residuals)
    terms = []

    for i in range(n):
        label = list("I" * n)
        label[n - 1 - i] = "Z"
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
    return np.array([int(b) for b in best])


def optimize_qaoa(
    cost_op: SparsePauliOp,
    reps: int = 2,
    maxiter: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Build and optimize a QAOA circuit for a given cost operator.

    Returns (best_bitstring, cost_history).
    """
    from scipy.optimize import minimize as scipy_minimize

    circuit = build_qaoa_circuit(cost_op, reps=reps)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)

    cost_history: list[float] = []

    def cost_fn(params):
        val = evaluate_qaoa_cost(params, circuit, cost_op)
        cost_history.append(val)
        return val

    result = scipy_minimize(cost_fn, x0, method="COBYLA", options={"maxiter": maxiter})
    labels = decode_qaoa_solution(result.x, circuit)
    return labels, cost_history
