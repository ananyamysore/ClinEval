"""Run a fully synthetic offline end-to-end ClinEval demonstration."""

from pathlib import Path

from clineval.pipeline import evaluate
from clineval.synthetic import generate_synthetic_predictions

if __name__=="__main__":
    out=Path("reports/synthetic-example")
    data,metadata=generate_synthetic_predictions(seed=2026,n_patients=600,repeated_measurements=2,miscalibrated=True)
    metadata["synthetic"]=True
    summary=evaluate(data,out,metadata=metadata,target_specificity=0.90,
                     subgroup_columns=["subgroup"],n_bootstrap=300,seed=2026)
    print(f"Synthetic-only report: {out.resolve() / 'report.md'}")
    print(f"Rows: {summary['n_records']}; audit checks: {len(summary['audit'])}")

