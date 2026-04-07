"""Tests for classical benchmarks."""

import numpy as np
import pytest

from quantum_anomaly_detection.classical.benchmarks import (
    run_isolation_forest,
    run_ocsvm,
    run_lof,
    run_dbscan,
    run_elliptic_envelope,
    run_simulated_annealing,
)


class TestClassicalMethods:
    def test_isolation_forest_shape(self, small_data):
        X, _ = small_data
        X_train, X_test = X[:40], X[40:]
        preds, scores = run_isolation_forest(X_train, X_test)
        assert preds.shape == (10,)
        assert scores.shape == (10,)

    def test_ocsvm_shape(self, small_data):
        X, _ = small_data
        X_train, X_test = X[:40], X[40:]
        preds, scores = run_ocsvm(X_train, X_test)
        assert preds.shape == (10,)

    def test_lof_shape(self, small_data):
        X, _ = small_data
        X_train, X_test = X[:40], X[40:]
        preds, scores = run_lof(X_train, X_test)
        assert preds.shape == (10,)

    def test_dbscan_output(self, small_data):
        X, _ = small_data
        labels = run_dbscan(X)
        assert labels.shape == (50,)

    def test_elliptic_envelope_shape(self, small_data):
        X, _ = small_data
        X_train, X_test = X[:40], X[40:]
        preds, scores = run_elliptic_envelope(X_train, X_test)
        assert preds.shape == (10,)


class TestSimulatedAnnealing:
    def test_sa_returns_binary_labels(self):
        D = np.array([[0, 1, 2, 3], [1, 0, 1.5, 2], [2, 1.5, 0, 1], [3, 2, 1, 0]])
        labels, history = run_simulated_annealing(D, n_iter=100, seed=42)
        assert set(labels).issubset({0, 1})
        assert len(labels) == 4
        assert len(history) > 0

    def test_sa_cost_decreases(self):
        """SA should reduce cost over iterations."""
        D = np.array([[0, 1, 5, 5], [1, 0, 5, 5], [5, 5, 0, 1], [5, 5, 1, 0]])
        D = D / D.max()
        labels, history = run_simulated_annealing(D, n_iter=2000, seed=42)
        # Best cost should be better than initial
        assert min(history) <= history[0]

    def test_sa_produces_balanced_split(self):
        """SA with balance penalty should produce a balanced 2-cluster split."""
        D = np.array([[0, 1, 2, 3], [1, 0, 1.5, 2], [2, 1.5, 0, 1], [3, 2, 1, 0]])
        D = D / D.max()
        labels, _ = run_simulated_annealing(D, n_iter=5000, seed=42)
        # Balance penalty should prevent all-one-cluster solutions
        n1 = labels.sum()
        assert 0 < n1 < len(labels)  # Not all same cluster
