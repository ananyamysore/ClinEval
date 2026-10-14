# Study-design assumptions and limitations

The analysis unit is determined by the study. Patient-grouped splitting prevents repeated observations from crossing partitions but does not fix site, family, temporal, or treatment leakage unless those constraints are declared. Temporal split helper preserves tied timestamps but is row-chronological and does not itself prevent repeated-patient overlap. Site/external-cohort evaluation can be represented as supplied partitions; no automatic site assignment is inferred.

Do not use the final test cohort to fit normalization, imputation, feature selection, calibration, or a threshold. Validation is for model/threshold selection; the final held-out test is for the frozen analysis. The audit can only inspect supplied prediction tables and metadata, not hidden model-training code. Missing metadata is marked not assessable, never evidence of no leakage.

ClinEval evaluates prediction outputs and does not estimate treatment effects, patient-specific recommendations, clinical utility, regulatory status, or prospective impact. Synthetic scenarios validate software behavior only and do not establish real-world clinical validity. Low-prevalence PPV, subgroup estimates, calibration, and threshold metrics require context-specific interpretation and external validation.

