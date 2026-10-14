"""Prediction-table validation and structured validation errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    """A machine-readable input validation issue."""

    code: str
    column: str | None
    message: str
    rows: tuple[int, ...] = ()


class DataValidationError(ValueError):
    """Raised when a prediction table cannot be safely evaluated."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("; ".join(f"{x.code}: {x.message}" for x in issues))

    def to_dict(self) -> list[dict[str, Any]]:
        """Return validation issues as JSON-serializable dictionaries."""
        return [asdict(issue) for issue in self.issues]


def validate_prediction_table(
    frame: pd.DataFrame,
    *,
    outcome: str = "y_true",
    probability: str = "y_prob",
    patient_id: str = "patient_id",
    record_id: str = "record_id",
    allow_missing_outcomes: bool = False,
    allow_invalid_probabilities: bool = False,
) -> pd.DataFrame:
    """Validate a long-form binary prediction table and return a defensive copy.

    ``allow_invalid_probabilities`` is intended for audit workflows: it preserves
    malformed scores so :func:`clineval.audit.audit_evaluation` can report them.
    Metric routines remain strict. Outcomes must be coded as 0/1; absent labels
    are rejected unless explicitly permitted.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError([ValidationIssue("empty_table", None, "input must be a nonempty DataFrame")])
    issues: list[ValidationIssue] = []
    for column in (outcome, probability):
        if column not in frame:
            issues.append(ValidationIssue("missing_column", column, f"required column {column!r} is absent"))
    if issues:
        raise DataValidationError(issues)
    out = frame.copy(deep=True)
    y = pd.to_numeric(out[outcome], errors="coerce")
    missing_y = y.isna()
    if missing_y.any() and not allow_missing_outcomes:
        issues.append(ValidationIssue("missing_outcome", outcome, "outcomes contain missing or nonnumeric values", tuple(np.flatnonzero(missing_y)[:20])))
    observed = y[~missing_y]
    if not observed.isin([0, 1]).all():
        issues.append(ValidationIssue("invalid_outcome", outcome, "binary outcomes must be coded as 0 or 1"))
    p = pd.to_numeric(out[probability], errors="coerce")
    bad_p = p.isna() | ~np.isfinite(p.astype(float)) | (p < 0) | (p > 1)
    if bad_p.any() and not allow_invalid_probabilities:
        issues.append(ValidationIssue("invalid_probability", probability, "probabilities must be finite and within [0, 1]", tuple(np.flatnonzero(bad_p)[:20])))
    if patient_id in out and out[patient_id].isna().any():
        issues.append(ValidationIssue("missing_patient_id", patient_id, "patient identifiers contain missing values", tuple(np.flatnonzero(out[patient_id].isna())[:20])))
    if record_id in out and out[record_id].duplicated().any():
        issues.append(ValidationIssue("duplicate_record_id", record_id, "record identifiers must be unique"))
    for key in ("split", "dataset_id", "model"):
        if key in out and out[key].isna().any():
            issues.append(ValidationIssue("missing_metadata", key, f"{key} contains missing values"))
    if issues:
        raise DataValidationError(issues)
    out[outcome] = y
    out[probability] = p
    return out

