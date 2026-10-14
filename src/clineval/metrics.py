"""Core binary classification metrics; independent of plotting and reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

def _arrays(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and y_prob must be nonempty matching 1D arrays")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    if np.isnan(y.astype(float)).any() or not np.isin(y, [0, 1]).all():
        raise ValueError("outcomes must be nonmissing binary values coded 0/1")
    return y.astype(int), p


def classification_metrics(y_true: Any, y_prob: Any, *, threshold: float = 0.5) -> dict[str, Any]:
    """Calculate discrimination and operating-point metrics for a binary endpoint.

    Average precision (stepwise recall-weighted precision) is returned separately
    from the trapezoidal PR curve area; they are not interchangeable. ROC-AUC is
    undefined with one outcome class and returned as ``None``. Predictive values
    depend on evaluation prevalence and should not be transported without adjustment.
    """
    y, p = _arrays(y_true, y_prob)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    pos = tp + fn
    neg = tn + fp
    denom_ppv = tp + fp
    denom_npv = tn + fn
    sensitivity = tp / pos if pos else None
    specificity = tn / neg if neg else None
    precision = tp / denom_ppv if denom_ppv else None
    npv = tn / denom_npv if denom_npv else None
    prevalence = float(y.mean())
    unique = np.unique(y)
    auc = float(roc_auc_score(y, p)) if len(unique) == 2 else None
    ap = float(average_precision_score(y, p)) if pos else None
    brier = float(np.mean((p - y) ** 2))
    return {
        "n": int(len(y)), "n_positive": int(pos), "n_negative": int(neg),
        "prevalence": prevalence, "threshold": float(threshold),
        "roc_auc": auc, "average_precision": ap,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision_ppv": precision, "negative_predictive_value": npv,
        "brier_score": brier,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "undefined": {"roc_auc": len(unique) < 2, "sensitivity": pos == 0,
                      "specificity": neg == 0, "precision_ppv": denom_ppv == 0,
                      "negative_predictive_value": denom_npv == 0},
    }

