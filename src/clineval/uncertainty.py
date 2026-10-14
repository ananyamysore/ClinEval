"""Bootstrap confidence intervals with independent-unit resampling."""

from __future__ import annotations

import numpy as np

from .metrics import classification_metrics, _arrays


def bootstrap_metric_ci(y_true, y_prob, *, metric: str = "roc_auc", confidence: float = 0.95,
                        n_resamples: int = 2000, seed: int = 0, groups=None,
                        threshold: float = 0.5) -> dict:
    """Percentile bootstrap CI; resample rows or independent clusters when supplied.

    Cluster bootstrap samples patients with replacement and includes all their
    rows. Intervals may be unstable with few independent units or rare outcomes;
    no CI is reported if insufficient finite replicates remain.
    """
    y, p = _arrays(y_true, y_prob)
    if not 0 < confidence < 1 or n_resamples < 100:
        raise ValueError("confidence must be in (0,1) and n_resamples >= 100")
    if metric not in {"roc_auc", "average_precision", "brier_score", "sensitivity", "specificity", "precision_ppv", "negative_predictive_value"}:
        raise ValueError(f"unsupported metric: {metric}")
    if groups is None:
        units = np.arange(len(y))
        indices_by_unit = {i: np.array([i]) for i in units}
        unit_method = "row_bootstrap_independent_rows_assumed"
    else:
        groups = np.asarray(groups)
        if groups.shape != y.shape or pd_is_missing(groups).any():
            raise ValueError("groups must be matching and nonmissing")
        units = np.unique(groups)
        indices_by_unit = {u: np.flatnonzero(groups == u) for u in units}
        unit_method = "cluster_bootstrap"
    if len(units) < 2:
        return {"estimate": None, "lower": None, "upper": None, "status": "inconclusive", "reason": "fewer than two independent units"}
    point = classification_metrics(y, p, threshold=threshold).get(metric)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        sampled = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([indices_by_unit[u] for u in sampled])
        value = classification_metrics(y[idx], p[idx], threshold=threshold).get(metric)
        if value is not None and np.isfinite(value):
            vals.append(value)
    if len(vals) < max(50, int(0.5*n_resamples)):
        return {"estimate": point, "lower": None, "upper": None, "status": "inconclusive", "reason": "too few finite bootstrap replicates", "finite_replicates": len(vals), "method": unit_method}
    alpha = (1-confidence)/2
    lo, hi = np.quantile(vals, [alpha, 1-alpha])
    return {"estimate": point, "lower": float(lo), "upper": float(hi), "confidence": confidence,
            "method": "percentile_bootstrap", "unit_method": unit_method,
            "n_independent_units": int(len(units)), "n_resamples": n_resamples,
            "finite_replicates": len(vals), "seed": seed, "status": "ok"}


def pd_is_missing(values):
    """Small local helper avoiding a pandas dependency in this module's public API."""
    import pandas as pd
    return pd.isna(values)

