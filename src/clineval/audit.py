"""Structured evaluation-validity checks; absence of provenance is not a clean bill."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AuditFinding:
    """One auditable issue, warning, or non-assessable check."""

    rule_id: str
    severity: str
    status: str
    explanation: str
    evidence: dict[str, Any]
    remediation: str


def audit_evaluation(frame: pd.DataFrame, *, metadata: dict[str, Any] | None = None,
                     forbidden_columns: list[str] | None = None,
                     group_key: str | None = None,
                     prediction_time_col: str = "prediction_time",
                     intended_time_col: str = "intended_prediction_time") -> list[dict[str, Any]]:
    """Run documented leakage and reporting checks on predictions and provenance.

    Findings distinguish confirmed violations (``status='violation'``), warnings,
    and checks that cannot be assessed. Feature names are weak leakage signals;
    the function never claims that unflagged features are leakage-free.
    """
    meta = metadata or {}
    findings: list[AuditFinding] = []
    def add(rule, severity, status, explanation, evidence=None, remediation=""):
        findings.append(AuditFinding(rule, severity, status, explanation, evidence or {}, remediation))

    split_key = str(meta.get("split_column", "split"))
    if split_key not in frame:
        add("AUDIT_SPLIT_PROVENANCE", "warning", "not_assessable", "Split labels are absent; patient overlap and split-specific checks cannot be completed.", {}, "Supply train/validation/test split labels and their provenance.")
        splits = None
    else:
        splits = frame[split_key].astype(str)
    patient_key = str(meta.get("patient_id_column", "patient_id"))
    if splits is not None and patient_key in frame:
        patient_sets = {part:set(frame.loc[splits == part, patient_key].dropna().astype(str)) for part in ("train", "validation", "test")}
        overlaps = {f"{a}_{b}": sorted(patient_sets[a]&patient_sets[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))}
        nonempty = {k:v for k,v in overlaps.items() if v}
        add("AUDIT_PATIENT_OVERLAP", "error" if nonempty else "info", "violation" if nonempty else "passed",
            "Patient identifiers are shared across partitions." if nonempty else "No shared patient identifiers were found across supplied partitions.",
            {k:{"n_overlap":len(v),"examples":v[:10]} for k,v in overlaps.items()},
            "Recreate partitions at patient level and keep all encounters from a patient in one partition." if nonempty else "Retain the split manifest for reproducibility.")
    else:
        add("AUDIT_PATIENT_OVERLAP", "warning", "not_assessable", "Patient identifiers or split labels are unavailable.", {}, "Supply patient identifiers and partition labels; missing provenance is not evidence of independence.")

    if group_key:
        if splits is None or group_key not in frame:
            add("AUDIT_GROUP_LEAKAGE", "warning", "not_assessable", "Declared grouping key or split labels are unavailable.", {"group_key":group_key}, "Supply the group key and split assignments.")
        else:
            nsplit=frame.dropna(subset=[group_key]).groupby(group_key)[split_key].nunique()
            leaked=nsplit[nsplit>1].index.astype(str).tolist()
            add("AUDIT_GROUP_LEAKAGE", "error" if leaked else "info", "violation" if leaked else "passed",
                "Declared groups occur in multiple partitions." if leaked else "Declared groups are partition-disjoint.",
                {"group_key":group_key,"n_leaked":len(leaked),"examples":leaked[:10]},
                "Split at the declared independent-group level." if leaked else "Document the grouping rule.")
    else:
        add("AUDIT_GROUP_LEAKAGE", "warning", "not_assessable", "No study-design group key was declared.", {}, "Declare patient, site, family, or other group restrictions required by the study design.")

    if splits is not None and prediction_time_col in frame:
        t=pd.to_datetime(frame[prediction_time_col],errors="coerce",utc=True)
        errors=int(t.isna().sum())
        order=[]
        for part in ("train","validation","test"):
            vals=t[splits==part].dropna()
            order.append((part,vals.max() if len(vals) else None))
        bad=any(order[i][1] is not None and order[i+1][1] is not None and order[i][1] >= order[i+1][1] for i in range(2))
        add("AUDIT_TEMPORAL_ORDER", "error" if bad or errors else "info", "violation" if bad or errors else "passed",
            "Partition timestamps violate declared train < validation < test chronology or are unparseable." if bad or errors else "Partition timestamps are ordered according to train < validation < test.",
            {"max_time_by_split":{k:(str(v) if v is not None else None) for k,v in order},"unparseable":errors},
            "Use forward-in-time partitions and verify all feature/label availability times.")
    else:
        add("AUDIT_TEMPORAL_ORDER", "warning", "not_assessable", "Prediction timestamps are unavailable.", {}, "Record prediction-time timestamps and declare temporal split requirements.")
    if prediction_time_col in frame and intended_time_col in frame:
        pred=pd.to_datetime(frame[prediction_time_col],errors="coerce",utc=True)
        intended=pd.to_datetime(frame[intended_time_col],errors="coerce",utc=True)
        later=pred>intended
        add("AUDIT_FEATURE_AFTER_INDEX", "error" if later.any() else "info", "violation" if later.any() else "passed",
            "Prediction timestamps occur after the intended prediction time." if later.any() else "No prediction timestamp is later than intended time.",
            {"n_rows":int(later.sum()),"examples":np.flatnonzero(later)[:10].tolist()}, "Exclude information recorded after the intended decision time.")
    else:
        add("AUDIT_FEATURE_AFTER_INDEX", "warning", "not_assessable", "Prediction and intended prediction timestamps are not both supplied.", {}, "Supply both timestamps and feature availability-time metadata.")

    if "record_id" in frame:
        dup=frame.record_id.duplicated(keep=False)
        add("AUDIT_DUPLICATE_RECORDS", "error" if dup.any() else "info", "violation" if dup.any() else "passed",
            "Duplicate record identifiers were found." if dup.any() else "Record identifiers are unique.", {"n_rows":int(dup.sum()),"examples":frame.loc[dup,"record_id"].astype(str).unique().tolist()[:10]}, "Resolve duplicate records or document legitimate repeated records with unique identifiers.")
    else:
        add("AUDIT_DUPLICATE_RECORDS", "warning", "not_assessable", "No record identifier was supplied.", {}, "Supply unique record identifiers.")
    candidate=[c for c in frame.columns if c not in {"y_true","y_prob",split_key,patient_key,"record_id","model","dataset_id"}]
    duplicated=frame.duplicated(subset=[c for c in ("y_true","y_prob",patient_key) if c in frame],keep=False) if all(c in frame for c in ("y_true","y_prob")) else pd.Series(False,index=frame.index)
    add("AUDIT_SUSPICIOUS_DUPLICATES", "warning" if duplicated.any() else "info", "warning" if duplicated.any() else "passed",
        "Repeated outcome/probability/patient tuples may represent duplicated predictions." if duplicated.any() else "No repeated outcome/probability/patient tuples detected.", {"n_rows":int(duplicated.sum()),"candidate_feature_columns":candidate[:20]}, "Review repeated prediction rows; exact equality is only a heuristic and may be legitimate.")

    forbidden=set(forbidden_columns or meta.get("forbidden_columns",[]))
    present=sorted(forbidden & set(frame.columns))
    obvious=[c for c in frame.columns if any(token in c.lower() for token in ("outcome_", "target_", "label_", "post_diagnosis", "discharge_status")) and c != "y_true"]
    detected=sorted(set(present+obvious))
    add("AUDIT_TARGET_LEAKAGE_INDICATORS", "error" if present else ("warning" if obvious else "info"), "violation" if present else ("warning" if obvious else "passed"),
        "Configured forbidden or target-derived feature columns are present." if detected else "No configured forbidden or obvious target-derived feature names were detected; arbitrary leakage cannot be ruled out.", {"configured_forbidden_present":present,"name_heuristic_flags":obvious}, "Remove target-derived fields and audit feature construction and timestamps. Do not rely on names alone.")

    scopes=meta.get("fit_scopes",{})
    for component in ("normalization","imputation","feature_selection","threshold_tuning","calibration"):
        scope=scopes.get(component)
        rule="AUDIT_"+component.upper()+"_SCOPE"
        if scope is None:
            add(rule,"warning","not_assessable",f"Fit scope for {component} is not recorded.",{},"Record the partition used to fit this component; fit on training/validation data as appropriate, never final test outcomes.")
        elif str(scope).lower()=="test":
            add(rule,"error","violation",f"{component} was fit using the final test set.",{"fit_scope":scope},"Refit without final test data and evaluate once on untouched test data.")
        else:
            add(rule,"info","passed",f"{component} fit scope is recorded as {scope}.",{"fit_scope":scope},"Retain pipeline provenance.")

    threshold_scope=meta.get("threshold_selected_on")
    if threshold_scope is None:
        add("AUDIT_THRESHOLD_LEAKAGE","warning","not_assessable","Threshold selection provenance is absent.",{},"Select thresholds on validation data and freeze before final test evaluation.")
    else:
        bad=str(threshold_scope).lower()=="test"
        add("AUDIT_THRESHOLD_LEAKAGE","error" if bad else "info","violation" if bad else "passed",
            "Threshold appears selected using test outcomes." if bad else f"Threshold selection partition is recorded as {threshold_scope}.",{"selected_on":threshold_scope},"Choose threshold on validation data, freeze it, then evaluate on test data.")
    calibration_scope=meta.get("calibration_fit_on")
    if calibration_scope is None:
        add("AUDIT_CALIBRATION_LEAKAGE","warning","not_assessable","Calibration-fit provenance is absent.",{},"Record calibration fit partition and keep it separate from final test evaluation.")
    else:
        bad=str(calibration_scope).lower()=="test"
        add("AUDIT_CALIBRATION_LEAKAGE","error" if bad else "info","violation" if bad else "passed",
            "Calibration was fit on final test outcomes." if bad else f"Calibration fit partition is recorded as {calibration_scope}.",{"fit_on":calibration_scope},"Fit calibration on training/validation data and evaluate calibration on untouched test data.")

    if "y_true" in frame:
        missing=int(frame.y_true.isna().sum())
        add("AUDIT_MISSING_OUTCOMES","error" if missing else "info","violation" if missing else "passed","Outcome values are missing." if missing else "Outcome values are complete.",{"n_missing":missing},"Resolve missing labels or define a prespecified missing-outcome handling plan.")
    if "y_prob" in frame:
        p=pd.to_numeric(frame.y_prob,errors="coerce").to_numpy(dtype=float)
        bad=~np.isfinite(p)|(p<0)|(p>1)
        add("AUDIT_INVALID_PROBABILITIES","error" if bad.any() else "info","violation" if bad.any() else "passed","Probabilities are nonfinite or outside [0,1]." if bad.any() else "Binary probabilities are finite and in [0,1].",{"n_invalid":int(bad.sum()),"examples":np.flatnonzero(bad)[:10].tolist()},"Fix model output transformation and validate probability semantics before evaluation.")
        multiclass=[c for c in frame if c.startswith("prob_")]
        if multiclass:
            q=frame[multiclass].apply(pd.to_numeric,errors="coerce").to_numpy(float)
            row_sums=np.nansum(q,axis=1)
            invalid=(~np.isfinite(q).all(axis=1))|(q<0).any(axis=1)|(q>1).any(axis=1)|(~np.isclose(row_sums,1,atol=1e-6))
            add("AUDIT_MULTICLASS_PROBABILITIES","error" if invalid.any() else "info","violation" if invalid.any() else "passed","Multiclass probability rows are malformed." if invalid.any() else "Multiclass probability rows are finite, bounded, and sum to one.",{"columns":multiclass,"n_invalid_rows":int(invalid.sum())},"Provide one finite probability per class summing to one.")

    missing_counts=not bool(meta.get("cohort_counts"))
    add("AUDIT_INCOMPLETE_REPORTING","warning" if missing_counts else "info","not_assessable" if missing_counts else "passed",
        "Cohort count provenance is absent; prevalence and subgroup denominators should be reported from eligible evaluation rows." if missing_counts else "Cohort count metadata is present.",{"has_cohort_counts":not missing_counts,"has_subgroup_columns":bool(meta.get("subgroup_columns"))},"Report eligible cohort counts, prevalence, subgroup denominators, and split metadata.")
    return [asdict(f) for f in findings]

