"""Pre-specified and validation-selected operating thresholds."""

from __future__ import annotations

import numpy as np

from .metrics import _arrays, classification_metrics


def evaluate_thresholds(y_true, y_prob, thresholds: list[float] | tuple[float, ...]) -> list[dict]:
    """Evaluate one or more predeclared thresholds on an evaluation cohort."""
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return [classification_metrics(y_true, y_prob, threshold=float(t)) for t in thresholds]


def threshold_for_target_specificity(y_true, y_prob, target_specificity: float) -> dict:
    """Select the lowest threshold achieving target specificity on supplied data.

    Ties are handled by evaluating unique observed scores. This function does not
    know the partition role; callers must pass validation data and separately
    evaluate the returned threshold on held-out test data.
    """
    y, p = _arrays(y_true, y_prob)
    if not 0 < target_specificity <= 1:
        raise ValueError("target_specificity must be in (0, 1]")
    if not (y == 0).any() or not (y == 1).any():
        return {"status": "inconclusive", "reason": "both outcome classes are required"}
    candidates = np.unique(np.r_[0.0, p, 1.0])
    valid = []
    for threshold in candidates:
        result = classification_metrics(y, p, threshold=float(threshold))
        if result["specificity"] is not None and result["specificity"] >= target_specificity:
            valid.append((float(threshold), result["sensitivity"], result["specificity"]))
    if not valid:
        return {"status": "inconclusive", "reason": "no observed-score threshold achieves target specificity"}
    # Among thresholds meeting specificity, maximize sensitivity; choose the smallest threshold on ties.
    best_sens = max(x[1] for x in valid)
    best = min((x for x in valid if x[1] == best_sens), key=lambda x: x[0])
    return {"status": "ok", "threshold": best[0], "sensitivity": best[1],
            "specificity": best[2], "target_specificity": float(target_specificity),
            "selection_rule": "maximize sensitivity subject to target specificity; ties choose lower threshold"}


def threshold_for_target_sensitivity(y_true, y_prob, target_sensitivity: float) -> dict:
    """Select the highest threshold achieving target sensitivity on supplied data."""
    y, p = _arrays(y_true, y_prob)
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    if not (y == 1).any() or not (y == 0).any():
        return {"status": "inconclusive", "reason": "both outcome classes are required"}
    candidates = np.unique(np.r_[0.0, p, 1.0])
    valid = []
    for t in candidates:
        m = classification_metrics(y, p, threshold=float(t))
        if m["sensitivity"] >= target_sensitivity:
            valid.append((float(t), m["sensitivity"], m["specificity"]))
    if not valid:
        return {"status": "inconclusive", "reason": "no observed-score threshold achieves target sensitivity"}
    best = max(valid, key=lambda x: x[0])
    return {"status": "ok", "threshold": best[0], "sensitivity": best[1],
            "specificity": best[2], "target_sensitivity": float(target_sensitivity),
            "selection_rule": "maximize specificity subject to target sensitivity"}

