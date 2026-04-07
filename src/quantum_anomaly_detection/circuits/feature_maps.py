"""Quantum feature maps for encoding classical data into quantum states.

Each feature map takes an n-dimensional classical vector x and produces an
n-qubit quantum state |phi(x)>. The choice of feature map determines the
geometry of the quantum feature space and affects kernel-based methods.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import zz_feature_map, pauli_feature_map


def build_zz_feature_map(
    n_qubits: int, reps: int = 2, entanglement: str = "full"
) -> QuantumCircuit:
    """ZZ feature map: Hadamard + single-qubit Z rotations + ZZ entangling rotations.

    Each layer applies H gates, then Rz(x_i) on each qubit, then Rzz(x_i * x_j)
    on entangled pairs. The 'full' entanglement connects all qubit pairs, making
    the kernel sensitive to all pairwise feature interactions.

    With 'full' entanglement and 2 reps, this creates a rich feature space that
    can capture non-linear correlations between input features.
    """
    return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)


def build_pauli_feature_map(
    n_qubits: int,
    reps: int = 2,
    paulis: list[str] | None = None,
    entanglement: str = "full",
) -> QuantumCircuit:
    """Generalized Pauli feature map with configurable rotation gates.

    Extends the ZZ feature map by allowing arbitrary Pauli rotation types
    (e.g., ["Z", "ZZ", "ZZZ"] for up to 3-body interactions). Higher-order
    Pauli terms capture more complex feature correlations at the cost of
    deeper circuits.
    """
    kwargs = {"reps": reps, "entanglement": entanglement}
    if paulis is not None:
        kwargs["paulis"] = paulis
    return pauli_feature_map(n_qubits, **kwargs)


def build_angle_encoding_map(n_qubits: int) -> QuantumCircuit:
    """Hand-built angle encoding feature map.

    Architecture: H -> Ry(x_i) -> linear CX chain -> Rz(x_i)

    The Hadamard layer creates superposition, Ry rotations encode feature
    values as angles on the Bloch sphere, CX gates entangle neighboring qubits,
    and final Rz rotations add a second encoding dimension per qubit.
    This is deliberately simple and transparent for educational purposes.
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="AngleEncoding")

    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits):
        qc.ry(x[i], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.rz(x[i], i)

    return qc


def assign_features(circuit: QuantumCircuit, x: np.ndarray) -> QuantumCircuit:
    """Bind a classical feature vector to a parameterized feature map circuit.

    Returns a fully bound circuit with no free parameters, ready for
    statevector simulation.
    """
    params = list(circuit.parameters)
    if len(params) != len(x):
        raise ValueError(f"Feature map has {len(params)} params, got {len(x)} features")
    param_dict = dict(zip(params, x))
    return circuit.assign_parameters(param_dict)
