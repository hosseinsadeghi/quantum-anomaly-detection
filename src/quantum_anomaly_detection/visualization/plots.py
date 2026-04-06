"""Visualization functions for anomaly detection results."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


def plot_anomaly_scores(
    scores: np.ndarray,
    y_true: np.ndarray | None = None,
    title: str = "Anomaly Scores",
    threshold: float | None = None,
) -> plt.Figure:
    """Histogram of anomaly scores, colored by true label if available."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if y_true is not None:
        ax.hist(scores[y_true == 0], bins=30, alpha=0.6, label="Normal", color="steelblue")
        ax.hist(scores[y_true == 1], bins=30, alpha=0.6, label="Anomaly", color="salmon")
        ax.legend()
    else:
        ax.hist(scores, bins=30, alpha=0.7, color="steelblue")

    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.3f}")
        ax.legend()

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_2d_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "Anomaly Detection",
    method_name: str = "",
) -> plt.Figure:
    """2D scatter plot with anomaly coloring. PCA to 2D if needed."""
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    # Normalize labels to 0/1
    lbl = np.array(labels).copy()
    if set(np.unique(lbl)).issubset({-1, 1}):
        lbl = (lbl == -1).astype(int)

    fig, ax = plt.subplots(figsize=(8, 6))
    normal = lbl == 0
    anomaly = lbl == 1

    ax.scatter(X_2d[normal, 0], X_2d[normal, 1], c="steelblue", s=20, alpha=0.6, label="Normal")
    ax.scatter(X_2d[anomaly, 0], X_2d[anomaly, 1], c="red", s=40, alpha=0.8, label="Anomaly", marker="x")

    full_title = f"{title} — {method_name}" if method_name else title
    ax.set_title(full_title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves",
) -> plt.Figure:
    """Overlay ROC curves for multiple methods.

    roc_data: {method_name: (y_true, scores)}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_true, scores) in roc_data.items():
        y = np.array(y_true).astype(int)
        s = np.array(scores)
        # Try both directions
        try:
            fpr, tpr, _ = roc_curve(y, s)
            roc_auc = auc(fpr, tpr)
            if roc_auc < 0.5:
                fpr, tpr, _ = roc_curve(y, -s)
                roc_auc = auc(fpr, tpr)
        except ValueError:
            continue
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_kernel_matrix(
    K: np.ndarray, title: str = "Quantum Kernel Matrix"
) -> plt.Figure:
    """Heatmap of kernel matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(K, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    return fig


def plot_optimization_history(
    cost_history: list[float], title: str = "Optimization History"
) -> plt.Figure:
    """Line plot of optimization cost vs iteration."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cost_history, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"
) -> plt.Figure:
    """Confusion matrix heatmap."""
    # Normalize predictions
    pred = np.array(y_pred).copy()
    if set(np.unique(pred)).issubset({-1, 1}):
        pred = (pred == -1).astype(int)

    cm = confusion_matrix(y_true.astype(int), pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_timeseries_anomalies(
    series: np.ndarray,
    anomaly_indices: np.ndarray,
    title: str = "Time Series Anomalies",
) -> plt.Figure:
    """Plot time series windows with anomalous ones highlighted."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot all windows as a continuous signal
    n_windows = len(series)
    for i in range(n_windows):
        start = i * len(series[i])
        t = np.arange(start, start + len(series[i]))
        color = "red" if i in anomaly_indices else "steelblue"
        alpha = 0.8 if i in anomaly_indices else 0.3
        ax.plot(t, series[i], color=color, alpha=alpha, linewidth=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", alpha=0.5, label="Normal"),
        Line2D([0], [0], color="red", alpha=0.8, label="Anomaly"),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    return fig
