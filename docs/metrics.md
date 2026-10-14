# Metrics and estimands

ClinEval currently implements binary classification. ROC-AUC estimates ranking discrimination across all thresholds; it is undefined if either outcome class is absent. Average precision is the recall-increment-weighted precision statistic (`sklearn.metrics.average_precision_score`). It is not the trapezoidal area under a precision-recall curve; no trapezoidal PR area is reported. At a frozen threshold, sensitivity is TP/(TP+FN), specificity is TN/(TN+FP), precision/PPV is TP/(TP+FP), and NPV is TN/(TN+FN). PPV and NPV depend strongly on evaluation prevalence.

Brier score is the mean squared probability error. Reliability bins summarize mean prediction and observed fraction positive; bin edges and strategy affect the result. Logistic calibration intercept and slope are descriptive recalibration summaries, not proof of transportability.

Target-specificity threshold selection searches unique observed validation scores and maximizes sensitivity subject to the target; tied sensitivity selects the lower threshold. The selected threshold must be frozen before the final test evaluation. ClinEval will use a prespecified threshold if validation is absent or selection is inconclusive, and will report that state.

Percentile bootstrap intervals resample rows only when independent rows are assumed. With patient identifiers, the pipeline samples patients and retains all their records. Few patients, rare outcomes, and undefined bootstrap replicates can make intervals unstable; the package withholds intervals when too few valid replicates remain. Bootstrap intervals do not address confounding, selection bias, or measurement error.

Subgroup estimates are exploratory point estimates with sample counts and patient-cluster bootstrap intervals. No multiplicity correction is applied. Subgroup differences are not established by point estimates or overlapping/non-overlapping intervals alone.

For multiple models, ClinEval compares Brier scores on common record IDs and reports an exploratory paired Wilcoxon test. This is not a replacement for a study-specific paired ROC procedure (e.g. DeLong) and row-level uncertainty can be inappropriate when records are clustered. Use patient-level resampling in the design-specific inferential analysis.

