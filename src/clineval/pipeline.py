"""End-to-end evaluation orchestration with validation-only threshold selection."""

from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from .audit import audit_evaluation
from .calibration import calibration_intercept_slope, calibration_summary
from .comparison import compare_models
from .data import validate_prediction_table
from .metrics import classification_metrics
from .reporting import write_evaluation_figures, write_json, write_tables
from .splitting import patient_group_split
from .subgroups import subgroup_metrics
from .thresholds import threshold_for_target_specificity
from .uncertainty import bootstrap_metric_ci


def evaluate(frame: pd.DataFrame, out_dir: str|Path, *, metadata: dict|None=None,
             threshold: float=0.5, target_specificity: float|None=None,
             subgroup_columns: list[str]|None=None, n_bootstrap: int=1000,
             seed: int=0) -> dict:
    """Run audits and binary model evaluation, exporting JSON/CSV/PNG artifacts.

    If ``split`` is absent, patient-grouped splits are generated when patient IDs
    exist; otherwise an error is raised rather than assuming rows are independent.
    Target-specific thresholds are selected only on validation data and assessed
    on test data. No threshold is optimized on final test outcomes.
    """
    meta=dict(metadata or {})
    data=frame.copy()
    if "split" not in data:
        if "patient_id" not in data:
            raise ValueError("split labels or patient_id are required; row independence is not assumed")
        split=patient_group_split(data,seed=seed)
        data["split"]=split.labels
        meta["generated_split"]={"method":split.metadata["method"],"seed":seed,"patient_counts":split.metadata["patient_counts"]}
    data=validate_prediction_table(data,allow_invalid_probabilities=False)
    meta.setdefault("cohort_counts", {"records_total":len(data),
                                      "records_by_split":data.split.value_counts().astype(int).to_dict()})
    if "model" not in data: data["model"]="model"
    if not set(data.split.astype(str)).issubset({"train","validation","test"}):
        raise ValueError("split values must be train, validation, and/or test")
    if target_specificity is not None and (data.split.astype(str)=="validation").any():
        meta.setdefault("threshold_selected_on", "validation")
    elif target_specificity is None:
        meta.setdefault("threshold_selected_on", "prespecified")
    audit=audit_evaluation(data,metadata=meta,group_key=meta.get("group_key"))
    warnings=[f for f in audit if f["severity"] in {"warning","error"} and f["status"]!="passed"]
    results=[]
    for model, model_frame in data.groupby("model",sort=True,observed=True):
        validation=model_frame[model_frame.split=="validation"]
        test=model_frame[model_frame.split=="test"]
        selected=float(threshold)
        selection={"status":"prespecified","threshold":selected,"selection_partition":None}
        if target_specificity is not None:
            if validation.empty:
                selection={"status":"inconclusive","reason":"validation partition is empty; using prespecified threshold","threshold":selected}
                warnings.append({"rule_id":"PIPELINE_NO_VALIDATION","severity":"warning","status":"not_assessable","explanation":"No validation set available for threshold selection.","evidence":{},"remediation":"Provide validation rows or use a prespecified threshold."})
            else:
                selection=threshold_for_target_specificity(validation.y_true,validation.y_prob,target_specificity)
                if selection["status"]=="ok":
                    selected=float(selection["threshold"]); selection["selection_partition"]="validation"
                else:
                    selection["threshold"]=selected
                    warnings.append({"rule_id":"PIPELINE_THRESHOLD_INCONCLUSIVE","severity":"warning","status":"not_assessable","explanation":selection.get("reason","threshold selection inconclusive"),"evidence":{},"remediation":"Provide both classes in validation data or use a prespecified threshold."})
        if test.empty:
            results.append({"model":str(model),"status":"inconclusive","reason":"test partition is empty","threshold_selection":selection})
            continue
        metrics=classification_metrics(test.y_true,test.y_prob,threshold=selected)
        group=test.patient_id if "patient_id" in test else None
        ci={m:bootstrap_metric_ci(test.y_true,test.y_prob,metric=m,n_resamples=n_bootstrap,
                                  seed=seed,groups=group,threshold=selected)
            for m in ("roc_auc","average_precision","brier_score")}
        calibration=calibration_summary(test.y_true,test.y_prob)
        calibration.update(calibration_intercept_slope(test.y_true,test.y_prob))
        results.append({"model":str(model),"status":"evaluated","threshold_selection":selection,
                        "test_metrics":metrics,"confidence_intervals":ci,"calibration":calibration})
    subgroups=[]
    for model, model_frame in data.groupby("model",sort=True,observed=True):
        test=model_frame[model_frame.split=="test"]
        if test.empty: continue
        model_result=next((r for r in results if r.get("model")==str(model)),{})
        selected=model_result.get("threshold_selection",{}).get("threshold",threshold)
        cols=subgroup_columns or meta.get("subgroup_columns",[])
        if cols: subgroups.extend([{**r,"model":str(model)} for r in subgroup_metrics(test,subgroup_columns=cols,threshold=float(selected),n_resamples=n_bootstrap,seed=seed)])
    comparisons=compare_models(data,n_resamples=n_bootstrap,seed=seed) if data.model.nunique()>1 else []
    root=Path(out_dir); root.mkdir(parents=True,exist_ok=True)
    files=write_tables(root,metrics=results,subgroups=subgroups,audit=audit,comparisons=comparisons)
    files.extend(write_evaluation_figures(data,root))
    summary={"package":"ClinEval","version":"0.1.0","synthetic_data":bool(meta.get("synthetic",False)),
             "n_records":len(data),"n_patients":int(data.patient_id.nunique()) if "patient_id" in data else None,
             "split_counts":data.split.value_counts().to_dict(),"software_versions":{"python":platform.python_version(),
                "numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
             "seed":seed,"models":results,"subgroups":subgroups,"audit":audit,
             "comparisons":comparisons,"warnings":warnings,"outputs":files,
             "interpretation":"Evaluation metrics do not establish clinical utility or real-world validity."}
    write_json(root/"report.json",summary); files.append(str(root/"report.json"))
    # report.json lists itself for a complete artifact manifest.
    summary["outputs"]=files; write_json(root/"report.json",summary)
    (root/"report.md").write_text(_markdown_summary(summary))
    files.append(str(root/"report.md")); summary["outputs"]=files; write_json(root/"report.json",summary)
    return summary


def _markdown_summary(summary: dict) -> str:
    """Create a concise human-readable summary with clear synthetic labeling."""
    label="SYNTHETIC EXAMPLE — NOT CLINICAL EVIDENCE" if summary["synthetic_data"] else "Evaluation report"
    lines=[f"# ClinEval {label}","",f"Records: {summary['n_records']}; patients: {summary['n_patients']}","",
           "## Model results","","| Model | Status | ROC-AUC | Average precision | Brier | Threshold |","|---|---:|---:|---:|---:|---:|"]
    for r in summary["models"]:
        m=r.get("test_metrics",{})
        lines.append(f"| {r['model']} | {r['status']} | {m.get('roc_auc')} | {m.get('average_precision')} | {m.get('brier_score')} | {m.get('threshold',r.get('threshold_selection',{}).get('threshold'))} |")
    lines += ["","## Audit findings","",f"{len(summary['audit'])} checks were recorded. Missing provenance is labeled not assessable; a passing check is limited to supplied data.","",
              "## Interpretation","",summary["interpretation"],"", "See `audit.json`, `metrics.json`, and figures for machine-readable details.",""]
    return "\n".join(lines)

