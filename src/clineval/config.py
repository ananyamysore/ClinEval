"""Strict JSON configuration loading for the command-line workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationConfig:
    """Validated CLI defaults; explicit CLI arguments may override these values."""

    threshold: float = 0.5
    target_specificity: float | None = None
    subgroup_columns: tuple[str, ...] = ()
    bootstrap_resamples: int = 1000
    seed: int = 0

    def __post_init__(self):
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if self.target_specificity is not None and not 0 < self.target_specificity <= 1:
            raise ValueError("target_specificity must be in (0, 1]")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be >=100")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


def load_config(path: str | Path) -> EvaluationConfig:
    """Load a JSON config and reject unknown keys or invalid values."""
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a JSON object")
    allowed = {"threshold", "target_specificity", "subgroup_columns", "bootstrap_resamples", "seed"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    if "subgroup_columns" in values:
        columns = values["subgroup_columns"]
        if not isinstance(columns, list) or not all(isinstance(x, str) and x for x in columns):
            raise ValueError("subgroup_columns must be a list of nonempty strings")
        values["subgroup_columns"] = tuple(columns)
    return EvaluationConfig(**values)
