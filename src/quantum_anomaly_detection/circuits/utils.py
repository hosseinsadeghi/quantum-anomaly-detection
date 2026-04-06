"""Circuit utility functions — statevector helpers, parameter binding, drawing."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector


def get_statevector(circuit: QuantumCircuit) -> Statevector:
    """Get statevector from a fully bound (no free parameters) circuit."""
    return Statevector(circuit)


def bind_parameters(
    circuit: QuantumCircuit,
    params: np.ndarray,
    param_prefix: str | None = None,
) -> QuantumCircuit:
    """Bind parameter values to a circuit.

    If param_prefix is given, only bind parameters whose name starts with that
    prefix (useful for binding encoder vs decoder params separately).
    """
    if param_prefix is not None:
        target_params = [p for p in circuit.parameters if p.name.startswith(param_prefix)]
        if len(target_params) != len(params):
            raise ValueError(
                f"Expected {len(target_params)} params for prefix '{param_prefix}', "
                f"got {len(params)}"
            )
        param_dict = dict(zip(target_params, params))
    else:
        all_params = sorted(circuit.parameters, key=lambda p: p.name)
        if len(all_params) != len(params):
            raise ValueError(
                f"Expected {len(all_params)} params, got {len(params)}"
            )
        param_dict = dict(zip(all_params, params))
    return circuit.assign_parameters(param_dict)


def count_parameters(circuit: QuantumCircuit) -> int:
    """Return number of free parameters in circuit."""
    return circuit.num_parameters


def draw_circuit(circuit: QuantumCircuit, style: str = "mpl", **kwargs):
    """Draw circuit. Returns matplotlib figure when style='mpl'."""
    return circuit.draw(style, **kwargs)
