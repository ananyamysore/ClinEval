"""Paired model comparisons on the same evaluation population."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .metrics import classification_metrics
from .uncertainty import bootstrap_metric_ci


def compare_models(frame: pd.DataFrame, *, model_col="model", record_col="record_id",
                   outcome_col="y_true", probability_col="y_prob", test_split="test",
                   n_resamples=1000, seed=0) -> list[dict]:
    """Compare paired Brier scores on common test records; exploratory paired test.

    ROC-AUC paired inference is intentionally not implemented; use a validated
    DeLong procedure or patient-cluster bootstrap suited to the study design.
    """
    required={model_col,record_col,outcome_col,probability_col}
    if not required.issubset(frame):
        raise ValueError(f"missing model-comparison columns: {sorted(required-set(frame))}")
    test=frame[frame["split"]==test_split] if "split" in frame else frame.copy()
    models=sorted(test[model_col].astype(str).unique())
    outputs=[]
    for i,a in enumerate(models):
        for b in models[i+1:]:
            keys=["dataset_id",record_col] if "dataset_id" in test else [record_col]
            aa=test[test[model_col].astype(str)==a].set_index(keys)
            bb=test[test[model_col].astype(str)==b].set_index(keys)
            common=aa.index.intersection(bb.index)
            if len(common)<2:
                outputs.append({"model_a":a,"model_b":b,"status":"inconclusive","reason":"fewer than two common evaluation records"})
                continue
            ya=aa.loc[common,outcome_col]; yb=bb.loc[common,outcome_col]
            if not ya.equals(yb):
                outputs.append({"model_a":a,"model_b":b,"status":"invalid","reason":"outcomes differ on common record identifiers"})
                continue
            pa=aa.loc[common,probability_col].to_numpy(float); pb=bb.loc[common,probability_col].to_numpy(float)
            d=(pa-ya.to_numpy(float))**2-(pb-ya.to_numpy(float))**2
            p=float(wilcoxon(d).pvalue) if np.any(d) else 1.0
            outputs.append({"model_a":a,"model_b":b,"status":"descriptive_paired","n_common":len(common),
                "brier_a":classification_metrics(ya,pa)["brier_score"],"brier_b":classification_metrics(yb,pb)["brier_score"],
                "mean_brier_difference_a_minus_b":float(d.mean()),"wilcoxon_p_value":p,
                "brier_a_ci":bootstrap_metric_ci(ya,pa,metric="brier_score",n_resamples=n_resamples,seed=seed),
                "brier_b_ci":bootstrap_metric_ci(yb,pb,metric="brier_score",n_resamples=n_resamples,seed=seed),
                "interpretation":"exploratory paired comparison; row bootstrap assumes independent records"})
    return outputs

