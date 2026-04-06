"""Quantum feature maps for data encoding into quantum states."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import zz_feature_map, pauli_feature_map


def build_zz_feature_map(
    n_qubits: int, reps: int = 2, entanglement: str = "full"
) -> QuantumCircuit:
    """Build ZZ feature map using Qiskit library function.

    Encodes classical data x as: U_ZZ(x) = prod_k [ H^n * exp(i * sum phi(x) ZZ) ]
    """
    return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)


def build_pauli_feature_map(
    n_qubits: int,
    reps: int = 2,
    paulis: list[str] | None = None,
    entanglement: str = "full",
) -> QuantumCircuit:
    """Build Pauli feature map using Qiskit library function.

    Generalizes ZZ feature map to arbitrary Pauli rotations.
    """
    kwargs = {"reps": reps, "entanglement": entanglement}
    if paulis is not None:
        kwargs["paulis"] = paulis
    return pauli_feature_map(n_qubits, **kwargs)


def build_angle_encoding_map(n_qubits: int) -> QuantumCircuit:
    """Hand-built angle encoding: H layer + Ry(x_i) on each qubit + CX entanglement.

    This is a simple, transparent feature map built gate-by-gate.
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="AngleEncoding")

    # Hadamard layer
    for i in range(n_qubits):
        qc.h(i)

    # Rotation layer
    for i in range(n_qubits):
        qc.ry(x[i], i)

    # Entanglement layer (linear CX chain)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # Second rotation layer
    for i in range(n_qubits):
        qc.rz(x[i], i)

    return qc


def assign_features(circuit: QuantumCircuit, x: np.ndarray) -> QuantumCircuit:
    """Bind feature vector x to a parameterized feature map circuit."""
    params = list(circuit.parameters)
    if len(params) != len(x):
        raise ValueError(f"Feature map has {len(params)} params, got {len(x)} features")
    param_dict = dict(zip(params, x))
    return circuit.assign_parameters(param_dict)
