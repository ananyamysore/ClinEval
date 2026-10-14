"""Patient-grouped, temporal, and independent-row split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitResult:
    """Partition labels and split diagnostics."""

    labels: pd.Series
    metadata: dict


def patient_group_split(frame: pd.DataFrame, *, patient_col: str = "patient_id",
                        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
                        seed: int = 0) -> SplitResult:
    """Assign whole patients to train/validation/test using deterministic group splits.

    This is not a stratified group splitter; class balance must be inspected after
    splitting. It rejects fewer than three independent patient groups.
    """
    if patient_col not in frame or frame[patient_col].isna().any():
        raise ValueError(f"nonmissing {patient_col!r} is required")
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("fractions must be three positive values summing to one")
    groups = frame[patient_col].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("at least three patients are required")
    # GroupShuffleSplit sizes are proportions of groups in the supplied subset.
    first = GroupShuffleSplit(n_splits=1, test_size=fractions[1] + fractions[2], random_state=seed)
    train_idx, remain_idx = next(first.split(np.zeros(len(frame)), groups=groups))
    remain_groups = groups[remain_idx]
    relative_test = fractions[2] / (fractions[1] + fractions[2])
    second = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed + 1)
    val_rel, test_rel = next(second.split(np.zeros(len(remain_idx)), groups=remain_groups))
    labels = np.full(len(frame), "", dtype=object)
    labels[train_idx] = "train"
    labels[remain_idx[val_rel]] = "validation"
    labels[remain_idx[test_rel]] = "test"
    if np.any(labels == ""):
        raise RuntimeError("split assignment was incomplete")
    check = pd.DataFrame({"patient": groups, "split": labels}).groupby("patient").split.nunique()
    if (check > 1).any():
        raise RuntimeError("internal error: patient appears in multiple partitions")
    return SplitResult(pd.Series(labels, index=frame.index, name="split"),
                       {"method": "patient_grouped", "seed": seed,
                        "patient_counts": pd.Series(labels).groupby(labels).size().to_dict(),
                        "n_patients": int(len(unique))})


def temporal_split(frame: pd.DataFrame, *, timestamp_col: str = "prediction_time",
                   fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> SplitResult:
    """Split rows chronologically; equal timestamps are kept in one partition."""
    if timestamp_col not in frame:
        raise ValueError(f"missing timestamp column {timestamp_col!r}")
    time = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    if time.isna().any():
        raise ValueError("timestamps must be parseable and nonmissing")
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("fractions must be three positive values summing to one")
    order = np.argsort(time.to_numpy(), kind="stable")
    sorted_times = time.iloc[order]
    n = len(frame)
    cuts = [int(round(n * fractions[0])), int(round(n * (fractions[0]+fractions[1])))]
    # Move boundaries to avoid splitting tied timestamps.
    while cuts[0] < n and cuts[0] > 0 and sorted_times.iloc[cuts[0]-1] == sorted_times.iloc[cuts[0]]:
        cuts[0] += 1
    while cuts[1] < n and cuts[1] > 0 and sorted_times.iloc[cuts[1]-1] == sorted_times.iloc[cuts[1]]:
        cuts[1] += 1
    if not 0 < cuts[0] < cuts[1] < n:
        raise ValueError("temporal boundaries cannot produce nonempty ordered partitions")
    labels = np.empty(n, dtype=object)
    labels[order[:cuts[0]]] = "train"
    labels[order[cuts[0]:cuts[1]]] = "validation"
    labels[order[cuts[1]:]] = "test"
    return SplitResult(pd.Series(labels, index=frame.index, name="split"),
                       {"method": "chronological", "timestamp_col": timestamp_col,
                        "boundary_times": [str(sorted_times.iloc[cuts[0]-1]), str(sorted_times.iloc[cuts[1]-1])]})


def independent_row_split(frame: pd.DataFrame, *, fractions=(0.6, 0.2, 0.2), seed=0) -> SplitResult:
    """Random row split for explicitly independent observations only."""
    if "patient_id" in frame and frame.patient_id.nunique() < len(frame):
        raise ValueError("repeated patient rows detected; use patient_group_split")
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("fractions must be three positive values summing to one")
    indices = np.arange(len(frame))
    train, other = train_test_split(indices, test_size=fractions[1]+fractions[2], random_state=seed)
    val, test = train_test_split(other, test_size=fractions[2]/(fractions[1]+fractions[2]), random_state=seed+1)
    labels = np.full(len(frame), "", dtype=object)
    labels[train], labels[val], labels[test] = "train", "validation", "test"
    return SplitResult(pd.Series(labels, index=frame.index, name="split"), {"method":"independent_rows", "seed":seed})

