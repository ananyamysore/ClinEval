# Evaluation audit rules

Each finding records `rule_id`, `severity`, `status`, explanation, evidence, and remediation. `violation` means the supplied evidence demonstrates a problem; `warning` means risk or a heuristic flag; `not_assessable` means required provenance was missing. A `passed` check only describes fields supplied to ClinEval and never certifies a model pipeline as leakage-free.

| Rule | Evidence | Test coverage |
|---|---|---|
| `AUDIT_PATIENT_OVERLAP` | Shared patient IDs across train/validation/test | Valid grouped control and injected train/test patient overlap |
| `AUDIT_GROUP_LEAKAGE` | Declared group key occurs in multiple partitions | Same controls; repeated measurements |
| `AUDIT_TEMPORAL_ORDER` | Max partition timestamps not ordered | Forward temporal control / reversed timestamp fixture |
| `AUDIT_FEATURE_AFTER_INDEX` | Prediction timestamp after intended prediction time | Injected timestamp defect |
| `AUDIT_FEATURE_AVAILABILITY` | Declared feature timestamp after intended time | Valid timestamp control and future-feature defect |
| `AUDIT_DUPLICATE_RECORDS` | Duplicate record IDs | Injected duplicate record ID |
| `AUDIT_SUSPICIOUS_DUPLICATES` | Repeated outcome/probability/patient tuple | Heuristic; may flag legitimate repeated data |
| `AUDIT_TARGET_LEAKAGE_INDICATORS` | Configured forbidden field or obvious target-derived name | Injected forbidden target copy; name check is heuristic |
| `AUDIT_*_SCOPE` | Fit provenance says `test` or is missing | Test-scope and missing-provenance cases |
| `AUDIT_THRESHOLD_LEAKAGE` | Threshold selection partition is test/unknown | Test-selected threshold and valid validation metadata |
| `AUDIT_CALIBRATION_LEAKAGE` | Calibration fit partition is test/unknown | Test-fit and valid validation metadata |
| `AUDIT_MISSING_OUTCOMES` | Missing `y_true` | Injected missing outcome |
| `AUDIT_INVALID_PROBABILITIES` | NaN, infinity, or outside [0,1] | Injected invalid probability values |
| `AUDIT_MULTICLASS_PROBABILITIES` | Optional `prob_*` rows malformed or not summing to 1 | Injected malformed probability matrix; binary metrics remain the supported evaluation scope |
| `AUDIT_INCOMPLETE_REPORTING` | Missing cohort count provenance | Metadata-present/absent states |

The unit suite includes a valid control that must pass applicable checks and intentionally flawed controls for patient overlap, group leakage, temporal order, future feature availability, duplicate rows/IDs, target-derived fields, test-set threshold/calibration/preprocessing fit provenance, missing labels, and invalid binary/multiclass probabilities. Missing split and fit provenance is checked as not assessable.

Name heuristics cannot identify arbitrary target leakage. Declared fit scopes and timestamps are self-reported metadata. For rigorous audits, retain immutable split manifests, feature lineage, label windows, and preprocessing pipeline provenance.

