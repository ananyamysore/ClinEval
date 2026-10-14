"""Command-line interface for local ClinEval data validation and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import EvaluationConfig, load_config
from .data import DataValidationError, validate_prediction_table
from .pipeline import evaluate


def main(argv: list[str]|None=None) -> int:
    """Run the ``clineval`` command."""
    parser=argparse.ArgumentParser(prog="clineval",description="Audit and evaluate binary prediction tables")
    sub=parser.add_subparsers(dest="command",required=True)
    check=sub.add_parser("validate",help="validate a CSV prediction table")
    check.add_argument("--input",required=True)
    run=sub.add_parser("evaluate",help="run a complete evaluation and export a report")
    run.add_argument("--input",required=True); run.add_argument("--out",required=True)
    run.add_argument("--metadata",help="JSON provenance and audit metadata file")
    run.add_argument("--config",help="validated JSON evaluation configuration")
    run.add_argument("--threshold",type=float)
    run.add_argument("--target-specificity",type=float)
    run.add_argument("--subgroups",nargs="*")
    run.add_argument("--bootstrap",type=int); run.add_argument("--seed",type=int)
    args=parser.parse_args(argv)
    try:
        frame=pd.read_csv(args.input)
        if args.command=="validate":
            validate_prediction_table(frame)
            print(f"Valid prediction table: {len(frame)} rows")
            return 0
        metadata=json.loads(Path(args.metadata).read_text()) if args.metadata else {}
        config=load_config(args.config) if args.config else EvaluationConfig()
        result=evaluate(frame,args.out,metadata=metadata,
                        threshold=args.threshold if args.threshold is not None else config.threshold,
                        target_specificity=args.target_specificity if args.target_specificity is not None else config.target_specificity,
                        subgroup_columns=args.subgroups if args.subgroups is not None else list(config.subgroup_columns),
                        n_bootstrap=args.bootstrap if args.bootstrap is not None else config.bootstrap_resamples,
                        seed=args.seed if args.seed is not None else config.seed)
        print(f"Report written to {Path(args.out).resolve() / 'report.json'}")
        print(f"Audit checks: {len(result['audit'])}; warnings/errors: {len(result['warnings'])}")
        return 0
    except (ValueError, OSError, json.JSONDecodeError, DataValidationError) as exc:
        parser.exit(2,f"clineval: error: {exc}\n")


if __name__=="__main__":
    raise SystemExit(main())

