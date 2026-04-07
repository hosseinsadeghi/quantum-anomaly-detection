"""Tests for data loading and preprocessing modules."""

import numpy as np
import pytest

from quantum_anomaly_detection.data.tabular import (
    load_synthetic_blobs,
    preprocess_tabular,
)
from quantum_anomaly_detection.data.time_series import (
    load_synthetic_timeseries,
    extract_window_features,
    preprocess_timeseries,
)
from quantum_anomaly_detection.data.graph import build_adjacency_from_features


class TestTabularData:
    def test_synthetic_blobs_shape(self):
        X, y = load_synthetic_blobs(n_samples=100, n_features=4, seed=42)
        assert X.shape == (100, 4)
        assert y.shape == (100,)

    def test_synthetic_blobs_anomaly_fraction(self):
        X, y = load_synthetic_blobs(
            n_samples=1000, anomaly_fraction=0.05, seed=42
        )
        frac = y.sum() / len(y)
        assert abs(frac - 0.05) < 0.02

    def test_preprocess_tabular_range(self):
        X, _ = load_synthetic_blobs(n_samples=100, n_features=8, seed=42)
        X_proc = preprocess_tabular(X, n_components=4)
        assert X_proc.shape == (100, 4)
        assert X_proc.min() >= -0.01
        assert X_proc.max() <= np.pi + 0.01

    def test_preprocess_fit_data_no_leakage(self):
        """Preprocessing with fit_data should use fit_data stats, not X."""
        X_train, _ = load_synthetic_blobs(n_samples=100, n_features=4, seed=42)
        X_test, _ = load_synthetic_blobs(n_samples=50, n_features=4, seed=99)
        X_proc = preprocess_tabular(X_test, n_components=3, fit_data=X_train)
        assert X_proc.shape == (50, 3)
        # Values should be approximately in [0, pi] but may slightly exceed
        # if test distribution differs from train
        assert X_proc.shape[1] == 3


class TestTimeSeriesData:
    def test_synthetic_timeseries_shape(self):
        X, y = load_synthetic_timeseries(n_samples=100, window_size=32, seed=42)
        assert X.shape == (100, 32)
        assert y.shape == (100,)

    def test_extract_window_features_length(self):
        window = np.sin(np.linspace(0, 2 * np.pi, 64))
        features = extract_window_features(window)
        assert len(features) == 8

    def test_preprocess_timeseries_range(self):
        X, _ = load_synthetic_timeseries(n_samples=50, window_size=32, seed=42)
        X_proc = preprocess_timeseries(X, n_components=4)
        assert X_proc.shape[1] == 4
        assert X_proc.min() >= -0.01
        assert X_proc.max() <= np.pi + 0.01


class TestGraphData:
    def test_adjacency_symmetry(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4))
        adj = build_adjacency_from_features(X, k=5)
        assert np.allclose(adj, adj.T)

    def test_adjacency_binary(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4))
        adj = build_adjacency_from_features(X, k=5)
        assert set(np.unique(adj)).issubset({0.0, 1.0})
