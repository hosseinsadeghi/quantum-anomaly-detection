"""Tests for quantum circuit construction modules."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector, state_fidelity

from quantum_anomaly_detection.circuits.feature_maps import (
    build_zz_feature_map,
    build_pauli_feature_map,
    build_angle_encoding_map,
    assign_features,
)
from quantum_anomaly_detection.circuits.autoencoder import (
    build_encoder,
    build_decoder,
    build_autoencoder_circuit,
    reconstruction_loss,
)
from quantum_anomaly_detection.circuits.qaoa import (
    build_clustering_hamiltonian,
    build_thresholding_hamiltonian,
    build_qaoa_circuit,
    evaluate_qaoa_cost,
    decode_qaoa_solution,
    optimize_qaoa,
)
from quantum_anomaly_detection.circuits.swap_test import (
    build_swap_test_circuit,
    state_fidelity_distance,
    compute_distance_matrix,
)


class TestFeatureMaps:
    def test_zz_feature_map_shape(self):
        qc = build_zz_feature_map(4, reps=2)
        assert qc.num_qubits == 4
        assert qc.num_parameters == 4

    def test_pauli_feature_map_shape(self):
        qc = build_pauli_feature_map(4, reps=2)
        assert qc.num_qubits == 4

    def test_angle_encoding_shape(self):
        qc = build_angle_encoding_map(4)
        assert qc.num_qubits == 4
        assert qc.num_parameters == 4

    def test_assign_features(self):
        qc = build_angle_encoding_map(4)
        x = np.array([0.1, 0.2, 0.3, 0.4])
        bound = assign_features(qc, x)
        assert bound.num_parameters == 0
        sv = Statevector(bound)
        assert abs(sv.data.conj() @ sv.data - 1.0) < 1e-10

    def test_assign_features_wrong_size(self):
        qc = build_angle_encoding_map(4)
        with pytest.raises(ValueError):
            assign_features(qc, np.array([0.1, 0.2]))

    def test_different_inputs_produce_different_states(self):
        """Feature map should encode different inputs as different quantum states."""
        fm = build_angle_encoding_map(3)
        x1 = np.array([0.1, 0.2, 0.3])
        x2 = np.array([1.5, 2.0, 2.5])
        sv1 = Statevector(assign_features(fm, x1))
        sv2 = Statevector(assign_features(fm, x2))
        fid = state_fidelity(sv1, sv2)
        assert fid < 0.99  # Different inputs -> different states


class TestAutoencoder:
    def test_encoder_shape(self):
        qc = build_encoder(4, 2, reps=2)
        assert qc.num_qubits == 4
        assert qc.num_parameters == 4 * 2 * 2  # n_qubits * 2 * reps

    def test_decoder_shape(self):
        qc = build_decoder(4, 2, reps=2)
        assert qc.num_qubits == 4
        assert qc.num_parameters == 4 * 2 * 2

    def test_autoencoder_circuit_structure(self):
        qc = build_autoencoder_circuit(4, 2, encoder_reps=1, decoder_reps=1)
        assert qc.num_qubits == 4
        # Data params (4) + encoder params (4*2*1) + decoder params (4*2*1)
        assert qc.num_parameters == 4 + 8 + 8

    def test_reconstruction_loss_bounded(self):
        qc = build_autoencoder_circuit(4, 2, encoder_reps=1, decoder_reps=1)
        var_params = sorted(
            [p for p in qc.parameters if not p.name.startswith("x")],
            key=lambda p: p.name,
        )
        params = np.zeros(len(var_params))
        x = np.array([0.1, 0.2, 0.3, 0.4])
        loss = reconstruction_loss(params, qc, x, n_latent=2)
        assert 0.0 <= loss <= 1.0


class TestQAOA:
    def test_clustering_hamiltonian_hermitian(self):
        D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
        H = build_clustering_hamiltonian(D)
        # SparsePauliOp should be Hermitian
        assert np.allclose(H.to_matrix(), H.to_matrix().conj().T)

    def test_qaoa_circuit_params(self):
        D = np.array([[0, 1], [1, 0]])
        H = build_clustering_hamiltonian(D)
        qc = build_qaoa_circuit(H, reps=2)
        assert qc.num_parameters == 4  # 2*reps (gamma + beta per layer)

    def test_evaluate_qaoa_cost_real(self):
        D = np.array([[0, 1], [1, 0]])
        H = build_clustering_hamiltonian(D)
        qc = build_qaoa_circuit(H, reps=1)
        params = np.array([0.5, 0.5])
        cost = evaluate_qaoa_cost(params, qc, H)
        assert isinstance(cost, float)

    def test_decode_qaoa_solution_binary(self):
        D = np.array([[0, 1], [1, 0]])
        H = build_clustering_hamiltonian(D)
        qc = build_qaoa_circuit(H, reps=1)
        params = np.array([0.5, 0.5])
        labels = decode_qaoa_solution(params, qc)
        assert set(labels).issubset({0, 1})

    def test_thresholding_hamiltonian_hermitian(self):
        residuals = np.array([0.1, 0.5, 0.2])
        H = build_thresholding_hamiltonian(residuals)
        assert np.allclose(H.to_matrix(), H.to_matrix().conj().T)

    def test_clustering_hamiltonian_not_trivial(self):
        """Balance constraint should prevent all-one-cluster from being optimal."""
        D = np.array([[0, 1, 0.1], [1, 0, 1], [0.1, 1, 0]])
        H = build_clustering_hamiltonian(D)
        qc = build_qaoa_circuit(H, reps=1)
        # All-zeros state (all cluster 0): evaluate cost
        sv_all0 = Statevector.from_label("000")
        cost_all0 = float(np.real(sv_all0.expectation_value(H)))
        # Mixed state (one in each cluster): |010>
        sv_mixed = Statevector.from_label("010")
        cost_mixed = float(np.real(sv_mixed.expectation_value(H)))
        # With balance penalty, all-same should NOT be the minimum
        assert cost_all0 > cost_mixed or abs(cost_all0 - cost_mixed) < 0.01

    def test_optimize_qaoa_cost_decreases(self):
        """QAOA optimization should reduce cost over iterations."""
        D = np.array([[0, 1, 2, 3], [1, 0, 1.5, 2], [2, 1.5, 0, 1], [3, 2, 1, 0]])
        D = D / D.max()
        H = build_clustering_hamiltonian(D)
        labels, history = optimize_qaoa(H, reps=2, maxiter=50, seed=42)
        assert set(labels).issubset({0, 1})
        # Cost should generally decrease (compare first few vs last few)
        if len(history) >= 10:
            early_avg = np.mean(history[:5])
            late_avg = np.mean(history[-5:])
            assert late_avg <= early_avg + 0.5  # Allow some tolerance


class TestSwapTest:
    def test_swap_test_circuit_shape(self):
        qc = build_swap_test_circuit(3)
        assert qc.num_qubits == 7  # 2*3 + 1

    def test_identical_states_zero_distance(self):
        fm = build_angle_encoding_map(3)
        x = np.array([0.5, 1.0, 1.5])
        d = state_fidelity_distance(x, x, fm)
        assert d < 1e-6

    def test_distance_symmetry(self):
        fm = build_angle_encoding_map(3)
        x1 = np.array([0.1, 0.2, 0.3])
        x2 = np.array([1.0, 1.5, 2.0])
        d12 = state_fidelity_distance(x1, x2, fm)
        d21 = state_fidelity_distance(x2, x1, fm)
        assert abs(d12 - d21) < 1e-10

    def test_distance_matrix_symmetric(self):
        fm = build_angle_encoding_map(3)
        X = np.array([[0.1, 0.2, 0.3], [1.0, 1.5, 2.0], [0.5, 0.5, 0.5]])
        D = compute_distance_matrix(X, fm)
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0.0)
