"""Time series data loading and preprocessing — synthetic sensor/ECG data."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from scipy import stats


def load_synthetic_timeseries(
    n_samples: int = 500,
    window_size: int = 64,
    anomaly_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time series with injected anomalous windows.

    Normal windows: sinusoidal signal with noise.
    Anomalous windows: spike anomalies, frequency shifts, or level shifts.
    """
    rng = np.random.default_rng(seed)
    n_anomaly = int(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomaly

    t = np.linspace(0, 2 * np.pi, window_size)

    # Normal windows: sin with varying phase and small noise
    windows_normal = []
    for _ in range(n_normal):
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.8, 1.2)
        noise = rng.normal(0, 0.1, window_size)
        w = np.sin(freq * t + phase) + noise
        windows_normal.append(w)

    # Anomalous windows: mix of anomaly types
    windows_anomaly = []
    for i in range(n_anomaly):
        atype = i % 3
        if atype == 0:
            # Spike anomaly
            w = np.sin(t + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 0.1, window_size)
            spike_pos = rng.integers(0, window_size, size=3)
            w[spike_pos] += rng.choice([-1, 1], size=3) * rng.uniform(3, 5, size=3)
        elif atype == 1:
            # Frequency shift
            w = np.sin(3.0 * t + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 0.1, window_size)
        else:
            # Level shift
            w = np.sin(t + rng.uniform(0, 2 * np.pi)) + 3.0 + rng.normal(0, 0.1, window_size)
        windows_anomaly.append(w)

    X = np.array(windows_normal + windows_anomaly)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Extract statistical features from a single time series window.

    Features: [mean, std, max, min, skewness, kurtosis, fft_mag_1, fft_mag_2]
    """
    fft_mags = np.abs(np.fft.rfft(window))[1:]  # Skip DC component
    top_fft = np.sort(fft_mags)[::-1]

    features = [
        np.mean(window),
        np.std(window),
        np.max(window),
        np.min(window),
        float(stats.skew(window)),
        float(stats.kurtosis(window)),
        top_fft[0] if len(top_fft) > 0 else 0.0,
        top_fft[1] if len(top_fft) > 1 else 0.0,
    ]
    return np.array(features)


def preprocess_timeseries(
    X: np.ndarray,
    n_components: int = 8,
    fit_data: np.ndarray | None = None,
) -> np.ndarray:
    """Extract features from windows, then PCA + scale to [0, pi].

    If X is 2D with shape (n_samples, window_size), extracts features first.
    If X is already feature vectors, just does PCA + scaling.
    """
    # Extract features if raw windows
    if X.ndim == 2 and X.shape[1] > n_components * 2:
        features = np.array([extract_window_features(w) for w in X])
    else:
        features = X

    source_raw = fit_data if fit_data is not None else X
    if source_raw.ndim == 2 and source_raw.shape[1] > n_components * 2:
        source = np.array([extract_window_features(w) for w in source_raw])
    else:
        source = source_raw

    scaler = StandardScaler()
    scaler.fit(source)
    features_scaled = scaler.transform(features)

    if n_components < features.shape[1]:
        pca = PCA(n_components=n_components)
        pca.fit(scaler.transform(source))
        features_scaled = pca.transform(features_scaled)
        source_for_minmax = pca.transform(scaler.transform(source))
    else:
        source_for_minmax = scaler.transform(source)

    minmax = MinMaxScaler(feature_range=(0, np.pi))
    minmax.fit(source_for_minmax)
    return minmax.transform(features_scaled)
