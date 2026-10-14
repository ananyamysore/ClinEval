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
            groups=None
            if "patient_id" in aa and "patient_id" in bb:
                ga=aa.loc[common,"patient_id"]; gb=bb.loc[common,"patient_id"]
                if not ga.equals(gb):
                    outputs.append({"model_a":a,"model_b":b,"status":"invalid","reason":"patient identifiers differ for common evaluation records"})
                    continue
                groups=(aa.loc[common,["dataset_id","patient_id"]].astype(str).agg("::".join,axis=1).to_numpy()
                        if "dataset_id" in aa else ga.to_numpy())
            pa=aa.loc[common,probability_col].to_numpy(float); pb=bb.loc[common,probability_col].to_numpy(float)
            d=(pa-ya.to_numpy(float))**2-(pb-ya.to_numpy(float))**2
            unit_diffs=pd.Series(d,index=groups if groups is not None else np.arange(len(d))).groupby(level=0).mean().to_numpy()
            p=float(wilcoxon(unit_diffs).pvalue) if np.any(unit_diffs) and len(unit_diffs)>1 else (1.0 if len(unit_diffs)>1 else None)
            rng=np.random.default_rng(seed)
            diff_samples=[]
            if len(unit_diffs)>1:
                for _ in range(n_resamples):
                    sampled=rng.choice(unit_diffs,size=len(unit_diffs),replace=True)
                    diff_samples.append(float(sampled.mean()))
                difference_ci={"lower":float(np.quantile(diff_samples,.025)),"upper":float(np.quantile(diff_samples,.975)),
                               "method":"paired percentile bootstrap by patient" if groups is not None else "paired row bootstrap; independent records assumed",
                               "n_independent_units":len(unit_diffs),"n_resamples":n_resamples,"seed":seed}
            else:
                difference_ci={"lower":None,"upper":None,"method":"inconclusive","n_independent_units":len(unit_diffs)}
            outputs.append({"model_a":a,"model_b":b,"status":"descriptive_paired","n_common":len(common),
                "brier_a":classification_metrics(ya,pa)["brier_score"],"brier_b":classification_metrics(yb,pb)["brier_score"],
                "mean_brier_difference_a_minus_b":float(d.mean()),"mean_unit_difference_a_minus_b":float(unit_diffs.mean()),
                "difference_ci":difference_ci,"wilcoxon_p_value":p,
                "brier_a_ci":bootstrap_metric_ci(ya,pa,metric="brier_score",n_resamples=n_resamples,seed=seed,groups=groups),
                "brier_b_ci":bootstrap_metric_ci(yb,pb,metric="brier_score",n_resamples=n_resamples,seed=seed,groups=groups),
                "interpretation":"exploratory paired comparison; patient clusters are the resampling and paired-test unit when patient IDs are supplied"})
    return outputs

