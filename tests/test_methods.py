"""Tests for quantum anomaly detection methods."""

import numpy as np
import pytest

from quantum_anomaly_detection.circuits.feature_maps import build_zz_feature_map
from quantum_anomaly_detection.methods.quantum_kernel import compute_kernel_matrix
from quantum_anomaly_detection.methods.vqc_autoencoder import (
    train_autoencoder,
    score_anomalies,
    detect_anomalies,
)
from quantum_anomaly_detection.methods.quantum_distance import knn_anomaly_score
from quantum_anomaly_detection.methods.qaoa_clustering import (
    run_qaoa_clustering,
    identify_anomaly_clusters,
)


class TestQuantumKernel:
    def test_kernel_matrix_symmetric(self, small_data):
        X, _ = small_data
        fm = build_zz_feature_map(4)
        K = compute_kernel_matrix(X[:10], fm)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_kernel_matrix_diagonal_ones(self, small_data):
        X, _ = small_data
        fm = build_zz_feature_map(4)
        K = compute_kernel_matrix(X[:10], fm)
        assert np.allclose(np.diag(K), 1.0, atol=1e-8)

    def test_kernel_matrix_psd(self, small_data):
        X, _ = small_data
        fm = build_zz_feature_map(4)
        K = compute_kernel_matrix(X[:10], fm)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > -1e-8)

    def test_identical_points_have_kernel_one(self):
        """K(x, x) = 1 for any x (self-fidelity is always 1)."""
        fm = build_zz_feature_map(3)
        from quantum_anomaly_detection.methods.quantum_kernel import compute_kernel_entry
        x = np.array([0.5, 1.0, 1.5])
        assert abs(compute_kernel_entry(x, x, fm) - 1.0) < 1e-6

    def test_rectangular_kernel_matrix(self, small_data):
        """K(X, Y) should have shape (len(X), len(Y))."""
        X, _ = small_data
        fm = build_zz_feature_map(4)
        K = compute_kernel_matrix(X[:5], fm, Y=X[5:10])
        assert K.shape == (5, 5)


class TestVQCAutoencoder:
    def test_training_runs(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, np.pi, size=(10, 4))
        params, circuit, history = train_autoencoder(
            X, n_latent=2, encoder_reps=1, decoder_reps=1, maxiter=10, seed=42
        )
        assert len(params) > 0
        assert len(history) > 0

    def test_scores_shape(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, np.pi, size=(10, 4))
        params, circuit, _ = train_autoencoder(
            X, n_latent=2, encoder_reps=1, decoder_reps=1, maxiter=5, seed=42
        )
        scores = score_anomalies(X, params, circuit, n_latent=2)
        assert scores.shape == (10,)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_detect_anomalies_output(self):
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        labels = detect_anomalies(scores, contamination=0.4)
        assert set(labels).issubset({0, 1})
        assert labels.sum() > 0


class TestQuantumDistance:
    def test_knn_score_shape(self):
        D = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
        scores = knn_anomaly_score(D, k=2)
        assert scores.shape == (3,)
        assert np.all(scores >= 0)

    def test_outlier_has_highest_knn_score(self):
        """A point far from others should have the highest k-NN anomaly score."""
        D = np.array([
            [0.0, 0.1, 0.1, 5.0],
            [0.1, 0.0, 0.1, 5.0],
            [0.1, 0.1, 0.0, 5.0],
            [5.0, 5.0, 5.0, 0.0],  # outlier
        ])
        scores = knn_anomaly_score(D, k=2)
        assert np.argmax(scores) == 3


class TestQAOAClustering:
    def test_clustering_returns_binary_labels(self, tiny_data):
        X, _ = tiny_data
        labels, history = run_qaoa_clustering(X, reps=1, maxiter=20, seed=42)
        assert set(labels).issubset({0, 1})
        assert len(labels) == len(X)

    def test_identify_anomaly_clusters(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
        X = np.random.default_rng(42).standard_normal((10, 4))
        anomalies = identify_anomaly_clusters(labels, X, min_cluster_fraction=0.25)
        # Cluster 1 has 2/10 = 0.2 < 0.25, so should be anomaly
        assert anomalies[5] == 1
        assert anomalies[6] == 1
