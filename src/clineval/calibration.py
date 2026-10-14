"""Calibration summaries and reliability-curve data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .metrics import _arrays


def calibration_summary(y_true, y_prob, *, n_bins: int = 10, strategy: str = "quantile") -> dict:
    """Return Brier score and reliability bins; no calibration model is fit."""
    y, p = _arrays(y_true, y_prob)
    if n_bins < 2 or strategy not in {"quantile", "uniform"}:
        raise ValueError("n_bins must be >=2; strategy must be quantile or uniform")
    observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    table = pd.DataFrame({"mean_predicted": predicted, "fraction_positive": observed})
    return {"brier_score": float(np.mean((p-y)**2)), "n_bins_requested": n_bins,
            "strategy": strategy, "reliability": table.to_dict(orient="records")}


def calibration_intercept_slope(y_true, y_prob) -> dict[str, float | None]:
    """Estimate calibration intercept/slope with logistic recalibration.

    Returns nulls if outcomes lack both classes or the optimization is unstable.
    These are descriptive estimates and can be biased in small samples.
    """
    from scipy.optimize import minimize
    from scipy.special import expit, logit
    y, p = _arrays(y_true, y_prob)
    if len(np.unique(y)) < 2:
        return {"intercept": None, "slope": None}
    eps = np.finfo(float).eps
    lp = logit(np.clip(p, eps, 1-eps))
    def objective(beta):
        q = np.clip(expit(beta[0] + beta[1]*lp), eps, 1-eps)
        return -float(np.sum(y*np.log(q)+(1-y)*np.log1p(-q)))
    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return {"intercept": float(fit.x[0]), "slope": float(fit.x[1])} if fit.success else {"intercept": None, "slope": None}

