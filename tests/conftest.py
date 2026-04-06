"""Shared test fixtures — small synthetic data for fast tests."""

import numpy as np
import pytest


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def small_data(seed):
    """Small synthetic dataset: 50 samples, 4 features, scaled to [0, pi]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.pi, size=(50, 4))
    y = np.concatenate([np.zeros(45), np.ones(5)])
    perm = rng.permutation(50)
    return X[perm], y[perm]


@pytest.fixture
def tiny_data(seed):
    """Tiny dataset for QAOA tests: 8 samples, 4 features."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.pi, size=(8, 4))
    y = np.concatenate([np.zeros(6), np.ones(2)])
    return X, y
