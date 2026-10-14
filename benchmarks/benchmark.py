"""Reproducible microbenchmark harness; prints measurements only when run."""

from __future__ import annotations

import platform
import time
import tracemalloc

from clineval.audit import audit_evaluation
from clineval.synthetic import generate_synthetic_predictions


def main():
    frame,metadata=generate_synthetic_predictions(seed=77,n_patients=10000,repeated_measurements=3)
    tracemalloc.start(); start=time.perf_counter()
    findings=audit_evaluation(frame,metadata=metadata,group_key="patient_id")
    elapsed=time.perf_counter()-start; _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    print({"python":platform.python_version(),"rows":len(frame),"audit_findings":len(findings),
           "runtime_seconds":elapsed,"python_peak_traced_bytes":peak,"seed":77,
           "limitations":"tracemalloc excludes most native NumPy/pandas allocations; machine-dependent."})


if __name__=="__main__": main()

