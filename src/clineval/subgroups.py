"""Prespecified subgroup metrics with transparent denominators and uncertainty."""

from __future__ import annotations

import pandas as pd

from .metrics import classification_metrics
from .uncertainty import bootstrap_metric_ci


def subgroup_metrics(frame: pd.DataFrame, *, subgroup_columns: list[str], outcome="y_true",
                     probability="y_prob", threshold=0.5, patient_id="patient_id",
                     min_n=20, ci_resamples=1000, seed=0) -> list[dict]:
    """Report point estimates, sample counts and optional cluster-bootstrap CIs.

    Subgroups are exploratory unless prespecified. No multiplicity adjustment is
    implied; small groups are retained and labeled with a caution status.
    """
    results=[]
    for col in subgroup_columns:
        if col not in frame:
            raise ValueError(f"missing subgroup column {col!r}")
        for level, sub in frame.groupby(col, dropna=False, observed=True):
            if sub[outcome].isna().any():
                sub=sub.dropna(subset=[outcome])
            metrics=classification_metrics(sub[outcome], sub[probability], threshold=threshold)
            auc_ci=bootstrap_metric_ci(sub[outcome], sub[probability], metric="roc_auc",
                                       n_resamples=max(100,ci_resamples), seed=seed,
                                       groups=sub[patient_id] if patient_id in sub else None)
            results.append({"subgroup":col,"level":str(level),"n":len(sub),
                            "n_positive":int(sub[outcome].sum()),"n_negative":int((1-sub[outcome]).sum()),
                            "metrics":metrics,"roc_auc_ci":auc_ci,
                            "status":"exploratory_small_sample" if len(sub)<min_n else "exploratory",
                            "multiplicity_adjustment":"not applied"})
    return results

