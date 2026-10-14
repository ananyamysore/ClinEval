from clineval.audit import audit_evaluation
from clineval.splitting import patient_group_split, temporal_split
from clineval.synthetic import generate_synthetic_predictions


def by_rule(findings): return {item["rule_id"]:item for item in findings}


def test_patient_group_split_keeps_repeated_measurements_together():
    frame,_=generate_synthetic_predictions(n_patients=90,repeated_measurements=3)
    result=patient_group_split(frame,seed=2)
    check=frame.assign(split=result.labels).groupby("patient_id").split.nunique()
    assert check.max()==1


def test_temporal_split_is_forward_ordered():
    import pandas as pd
    frame=pd.DataFrame({"prediction_time":pd.date_range("2020-01-01",periods=30,tz="UTC")})
    labels=temporal_split(frame).labels
    assert list(labels.iloc[:18].unique())==["train"]
    assert labels.iloc[-1]=="test"


def test_valid_control_passes_overlap_and_known_defects_are_detected():
    valid,meta=generate_synthetic_predictions(seed=3,n_patients=90)
    findings=by_rule(audit_evaluation(valid,metadata=meta,group_key="patient_id"))
    for rule in ("AUDIT_PATIENT_OVERLAP","AUDIT_GROUP_LEAKAGE","AUDIT_TEMPORAL_ORDER",
                 "AUDIT_FEATURE_AFTER_INDEX","AUDIT_DUPLICATE_RECORDS",
                 "AUDIT_SUSPICIOUS_DUPLICATES","AUDIT_FEATURE_AVAILABILITY",
                 "AUDIT_TARGET_LEAKAGE_INDICATORS","AUDIT_THRESHOLD_LEAKAGE",
                 "AUDIT_CALIBRATION_LEAKAGE","AUDIT_MISSING_OUTCOMES",
                 "AUDIT_INVALID_PROBABILITIES","AUDIT_INCOMPLETE_REPORTING"):
        assert findings[rule]["status"]=="passed", rule
    defects={"patient_overlap","temporal_leakage","temporal_order_violation","future_feature_time",
             "invalid_probabilities","malformed_multiclass","missing_labels","test_threshold",
             "test_calibration","test_preprocessing_fit","target_leakage","duplicate_records",
             "suspicious_duplicates","incomplete_reporting"}
    bad,bad_meta=generate_synthetic_predictions(seed=4,n_patients=90,defects=defects)
    findings=by_rule(audit_evaluation(bad,metadata=bad_meta,group_key="patient_id"))
    for rule in ("AUDIT_PATIENT_OVERLAP","AUDIT_GROUP_LEAKAGE","AUDIT_TEMPORAL_ORDER","AUDIT_FEATURE_AFTER_INDEX",
                 "AUDIT_DUPLICATE_RECORDS","AUDIT_TARGET_LEAKAGE_INDICATORS","AUDIT_THRESHOLD_LEAKAGE",
                 "AUDIT_MISSING_OUTCOMES","AUDIT_INVALID_PROBABILITIES","AUDIT_FEATURE_AVAILABILITY",
                 "AUDIT_MULTICLASS_PROBABILITIES","AUDIT_NORMALIZATION_SCOPE",
                 "AUDIT_CALIBRATION_SCOPE","AUDIT_CALIBRATION_LEAKAGE"):
        assert findings[rule]["status"]=="violation", rule
    assert findings["AUDIT_SUSPICIOUS_DUPLICATES"]["status"]=="warning"
    assert findings["AUDIT_INCOMPLETE_REPORTING"]["status"]=="not_assessable"


def test_missing_provenance_is_not_assessed_as_passed():
    frame,_=generate_synthetic_predictions(n_patients=60)
    findings=by_rule(audit_evaluation(frame.drop(columns="patient_id"),metadata={}))
    assert findings["AUDIT_PATIENT_OVERLAP"]["status"]=="not_assessable"
    assert findings["AUDIT_THRESHOLD_LEAKAGE"]["status"]=="not_assessable"
    no_split,_=generate_synthetic_predictions(n_patients=60)
    findings=by_rule(audit_evaluation(no_split.drop(columns="split"),metadata={},group_key="patient_id"))
    assert findings["AUDIT_SPLIT_PROVENANCE"]["status"]=="not_assessable"


def test_chronology_is_not_assumed_without_temporal_design_declaration():
    frame,metadata=generate_synthetic_predictions(n_patients=60)
    metadata.pop("temporal_order_required")
    findings=by_rule(audit_evaluation(frame,metadata=metadata,group_key="patient_id"))
    assert findings["AUDIT_TEMPORAL_ORDER"]["status"]=="not_assessable"

