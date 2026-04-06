"""Evaluation metrics for anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict:
    """Compute anomaly detection metrics.

    y_true: ground truth (1 = anomaly, 0 = normal)
    y_pred: predicted labels (1 = anomaly, 0 = normal)
        Note: some methods use -1 for anomaly; this is handled automatically.
    scores: anomaly scores (higher = more anomalous), optional for AUC.
    """
    # Normalize predictions: convert -1/+1 to 1/0
    pred = np.array(y_pred).copy()
    if set(np.unique(pred)).issubset({-1, 1}):
        pred = (pred == -1).astype(int)

    true = np.array(y_true).astype(int)

    result = {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
    }

    if scores is not None and len(np.unique(true)) > 1:
        # For methods where lower score = anomaly, negate
        s = np.array(scores)
        try:
            result["auc_roc"] = float(roc_auc_score(true, s))
        except ValueError:
            # If score direction is wrong, try negating
            try:
                result["auc_roc"] = float(roc_auc_score(true, -s))
            except ValueError:
                result["auc_roc"] = float("nan")
        try:
            result["auc_pr"] = float(average_precision_score(true, s))
        except ValueError:
            result["auc_pr"] = float("nan")

    return result


def format_results_table(results: dict[str, dict]) -> pd.DataFrame:
    """Format multiple method results into a comparison DataFrame.

    results: {method_name: metrics_dict}
    """
    df = pd.DataFrame(results).T
    df.index.name = "Method"
    return df.round(4)
