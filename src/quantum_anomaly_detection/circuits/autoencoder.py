"""Variational Quantum Circuit Autoencoder — trash qubit approach.

The autoencoder compresses n qubits into n_latent qubits. The remaining
(n - n_latent) qubits are "trash" qubits. If compression is good, the trash
qubits end up in state |0...0>. Reconstruction loss = 1 - P(trash = |0...0>).
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace


def build_encoder(n_qubits: int, n_latent: int, reps: int = 2) -> QuantumCircuit:
    """Build parameterized encoder circuit.

    Uses Ry + Rz rotation layers with linear CX entanglement.
    Parameters are named 'enc_0', 'enc_1', ...
    """
    n_params = n_qubits * 2 * reps
    theta = ParameterVector("enc", n_params)
    qc = QuantumCircuit(n_qubits, name="Encoder")

    idx = 0
    for r in range(reps):
        for i in range(n_qubits):
            qc.ry(theta[idx], i)
            idx += 1
        for i in range(n_qubits):
            qc.rz(theta[idx], i)
            idx += 1
        # Entanglement: linear chain
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()

    return qc


def build_decoder(n_qubits: int, n_latent: int, reps: int = 2) -> QuantumCircuit:
    """Build parameterized decoder circuit (mirror structure of encoder).

    Parameters are named 'dec_0', 'dec_1', ...
    """
    n_params = n_qubits * 2 * reps
    theta = ParameterVector("dec", n_params)
    qc = QuantumCircuit(n_qubits, name="Decoder")

    idx = 0
    for r in range(reps):
        # Reverse entanglement
        for i in range(n_qubits - 2, -1, -1):
            qc.cx(i, i + 1)
        for i in range(n_qubits):
            qc.rz(theta[idx], i)
            idx += 1
        for i in range(n_qubits):
            qc.ry(theta[idx], i)
            idx += 1
        qc.barrier()

    return qc


def build_autoencoder_circuit(
    n_qubits: int,
    n_latent: int,
    encoder_reps: int = 2,
    decoder_reps: int = 2,
) -> QuantumCircuit:
    """Build full autoencoder: data_encoding + encoder + decoder.

    The circuit has three parameter groups:
    - 'x_*': data encoding parameters (n_qubits)
    - 'enc_*': encoder parameters
    - 'dec_*': decoder parameters

    Trash qubits are the last (n_qubits - n_latent) qubits.
    """
    x = ParameterVector("x", n_qubits)

    # Data encoding layer (angle encoding)
    qc = QuantumCircuit(n_qubits, name="QAutoencoder")
    for i in range(n_qubits):
        qc.ry(x[i], i)
    qc.barrier()

    # Encoder
    encoder = build_encoder(n_qubits, n_latent, encoder_reps)
    qc.compose(encoder, inplace=True)

    # Decoder
    decoder = build_decoder(n_qubits, n_latent, decoder_reps)
    qc.compose(decoder, inplace=True)

    return qc


def reconstruction_loss(
    params: np.ndarray,
    circuit: QuantumCircuit,
    x: np.ndarray,
    n_latent: int,
) -> float:
    """Compute reconstruction loss for one sample.

    Loss = 1 - P(trash qubits in |0...0>)

    The trash qubits are the last (n_qubits - n_latent) qubits.
    """
    n_qubits = circuit.num_qubits
    n_trash = n_qubits - n_latent

    # Separate data params from variational params
    x_params = sorted(
        [p for p in circuit.parameters if p.name.startswith("x")],
        key=lambda p: p.name,
    )
    var_params = sorted(
        [p for p in circuit.parameters if not p.name.startswith("x")],
        key=lambda p: p.name,
    )

    param_dict = {}
    for p, v in zip(x_params, x):
        param_dict[p] = v
    for p, v in zip(var_params, params):
        param_dict[p] = v

    bound = circuit.assign_parameters(param_dict)
    sv = Statevector(bound)

    # Trace out latent qubits (keep trash qubits)
    # Qiskit uses little-endian: qubit 0 is rightmost
    # Trash qubits are indices [n_latent, ..., n_qubits-1]
    # We trace out the latent qubits [0, ..., n_latent-1]
    latent_indices = list(range(n_latent))
    rho_trash = partial_trace(sv, latent_indices)

    # Probability of |0...0> on trash qubits
    zero_state = np.zeros(2**n_trash)
    zero_state[0] = 1.0
    prob_zero = np.real(zero_state @ rho_trash.data @ zero_state)

    return 1.0 - prob_zero


def batch_reconstruction_loss(
    params: np.ndarray,
    circuit: QuantumCircuit,
    X: np.ndarray,
    n_latent: int,
) -> float:
    """Mean reconstruction loss over a batch of samples."""
    losses = [reconstruction_loss(params, circuit, x, n_latent) for x in X]
    return float(np.mean(losses))
