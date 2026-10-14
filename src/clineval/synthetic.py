"""Deterministic synthetic prediction data for software validation only."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFECTS={"patient_overlap","temporal_leakage","invalid_probabilities","missing_labels","test_threshold","target_leakage","duplicate_records"}


def generate_synthetic_predictions(*, seed: int=0, n_patients: int=600,
                                   repeated_measurements: int=2,
                                   defects: set[str] | None=None,
                                   low_prevalence: bool=False,
                                   miscalibrated: bool=False) -> tuple[pd.DataFrame,dict]:
    """Generate grouped binary predictions with controllable known audit defects.

    Simulated labels and predictions are purely pedagogical. They are not evidence
    of clinical performance. Patients are assigned as units in the valid control.
    """
    active=set(defects or ())
    unknown=active-DEFECTS
    if unknown: raise ValueError(f"unknown synthetic defects: {sorted(unknown)}")
    if n_patients<30 or repeated_measurements<1: raise ValueError("need >=30 patients and >=1 measurement")
    rng=np.random.default_rng(seed)
    groups=np.array(["A" if i%2==0 else "B" for i in range(n_patients)])
    latent=rng.normal(size=n_patients)
    prevalence=0.07 if low_prevalence else 0.3
    prob_y=1/(1+np.exp(-(-2.55+1.2*latent))) if low_prevalence else 1/(1+np.exp(-(-0.3+0.9*latent)))
    patient_y=rng.binomial(1,prob_y)
    split=np.empty(n_patients,dtype=object)
    split[:int(.6*n_patients)]="train"; split[int(.6*n_patients):int(.8*n_patients)]="validation"; split[int(.8*n_patients):]="test"
    rng.shuffle(split)
    rows=[]
    base=pd.Timestamp("2022-01-01",tz="UTC")
    for i in range(n_patients):
        for visit in range(repeated_measurements):
            y=int(patient_y[i])
            # A subgroup-specific score attenuation creates a measurable disparity.
            signal=latent[i]*(0.55 if groups[i]=="B" else 1.0)
            score=1/(1+np.exp(-(-1.1+2.1*signal+1.2*(y-.5)+rng.normal(0,.55))))
            if miscalibrated: score=float(np.clip(score**0.45,0,1))
            split_rank={"train":0,"validation":1,"test":2}[str(split[i])]
            timestamp=base+pd.Timedelta(days=int(split_rank*n_patients*2+i*2+visit))
            rows.append({"record_id":f"r{i}_{visit}","patient_id":f"p{i}","dataset_id":"synthetic-demo",
                         "model":"model_demo","split":split[i],"y_true":y,"y_prob":score,
                         "subgroup":groups[i],"prediction_time":timestamp,
                         "intended_prediction_time":timestamp-pd.Timedelta(days=1) if "temporal_leakage" in active else timestamp})
    frame=pd.DataFrame(rows)
    if "patient_overlap" in active:
        # Copy a test record into train, retaining the patient ID to create a clear violation.
        row=frame.loc[frame.split=="test"].iloc[0].copy(); row["record_id"]="overlap-copy"; row["split"]="train"
        frame=pd.concat([frame,pd.DataFrame([row])],ignore_index=True)
    if "invalid_probabilities" in active:
        frame.loc[0,"y_prob"]=np.nan; frame.loc[1,"y_prob"]=1.4
    if "missing_labels" in active: frame.loc[2,"y_true"]=np.nan
    if "target_leakage" in active: frame["target_outcome_copy"]=frame.y_true
    if "duplicate_records" in active: frame.loc[1,"record_id"]=frame.loc[0,"record_id"]
    metadata={"patient_id_column":"patient_id","split_column":"split","group_key":"patient_id",
              "threshold_selected_on":"test" if "test_threshold" in active else "validation",
              "calibration_fit_on":"validation","fit_scopes":{"normalization":"train","imputation":"train","feature_selection":"train","threshold_tuning":"validation","calibration":"validation"},
              "forbidden_columns":["target_outcome_copy"] if "target_leakage" in active else [],
              "subgroup_columns":["subgroup"],"cohort_counts":{"total":len(frame)}}
    return frame,metadata

