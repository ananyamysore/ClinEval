# ClinEval

ClinEval is a local, CPU-only Python package for transparent evaluation and validity auditing of clinical and biomedical prediction models. It is an evaluation library—not clinical decision support. It does not make patient-specific predictions or claim regulatory approval. Synthetic examples are software fixtures, not evidence of clinical validity.

## Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,lint]"
```

## Quick start

```bash
python examples/run_synthetic.py
```

Or evaluate your own CSV with columns `y_true`, `y_prob`, `patient_id`, `record_id`, `split` (`train`, `validation`, `test`), and optional `model`, `dataset_id`, subgroup columns and time fields:

```bash
clineval validate --input predictions.csv
clineval evaluate --input predictions.csv --out reports/run-001 \
  --metadata configs/example-metadata.json --config configs/example.json
```

Thresholds for a target operating point are selected only on the validation split and are assessed on held-out test rows. A prespecified `--threshold` is used when no target is requested. If split labels are absent, ClinEval uses patient-grouped splitting when patient IDs exist; it refuses to assume that rows are independent.

## What is included

- Strict binary prediction-table validation with structured errors.
- ROC-AUC, average precision, confusion matrix, sensitivity, specificity, PPV, NPV and Brier score. Undefined quantities remain null/undefined rather than being silently set to zero.
- Reliability curve data and calibration intercept/slope summaries.
- Validation-only target-specificity threshold selection.
- Row bootstrap and patient-cluster bootstrap confidence intervals.
- Exploratory subgroup metrics with denominators and explicit lack of multiplicity correction.
- Patient/group overlap, chronology, duplicate, target-leakage indicator, fit-scope, threshold/calibration provenance, missing-label, and probability audits.
- Paired common-population Brier comparison for multiple models.
- JSON and CSV outputs, a Markdown summary, and ROC/PR/calibration figures.
- Deterministic valid and intentionally flawed synthetic fixtures, offline examples, tests, CI and a runtime/memory benchmark harness.

See [metrics](docs/metrics.md), [audit rules](docs/audit-rules.md), and [study-design assumptions](docs/study-design.md). [Sample report instructions](docs/sample-report.md) generate clearly labeled synthetic output at runtime.

## Tests, lint, benchmark

```bash
python -m pytest -q
python -m ruff check src tests examples benchmarks
python benchmarks/benchmark.py
```

No benchmark numbers are claimed in advance; the benchmark prints machine-specific observations when run. The bootstrap test intervals and point estimates are exploratory and do not establish clinical utility. Missing provenance is reported as not assessable. Audits cannot inspect hidden training code or reliably find arbitrary leakage from feature names.

