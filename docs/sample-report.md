# Synthetic sample report

The CI workflow runs `python examples/run_synthetic.py` and prints the generated report. The example uses a deterministic synthetic fixture and writes machine-readable JSON/CSV tables plus ROC, precision-recall, and calibration figures under `reports/synthetic-example/`. Results are explicitly labeled synthetic and are not clinical evidence. They validate software execution only, not real-world performance.

