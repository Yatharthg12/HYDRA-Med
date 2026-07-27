"""Deterministic patient-grouped train/validation/test splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ExperimentConfig


def create_patient_splits(
    frame: pd.DataFrame, config: ExperimentConfig
) -> pd.Series:
    """Assign complete patient groups to a deterministic 70/15/15 split."""
    patient_target = frame.groupby("patient_nbr", sort=True)["target"].max()
    patients = patient_target.index.to_numpy()
    strata = patient_target.to_numpy()
    train_patients, remaining = train_test_split(
        patients,
        test_size=config.validation_ratio + config.test_ratio,
        random_state=config.random_seed,
        stratify=strata,
    )
    remaining_target = patient_target.loc[remaining].to_numpy()
    validation_fraction = config.validation_ratio / (
        config.validation_ratio + config.test_ratio
    )
    validation_patients, test_patients = train_test_split(
        remaining,
        train_size=validation_fraction,
        random_state=config.random_seed,
        stratify=remaining_target,
    )
    mapping = {str(patient): "train" for patient in train_patients}
    mapping.update({str(patient): "validation" for patient in validation_patients})
    mapping.update({str(patient): "test" for patient in test_patients})
    assignments = frame["patient_nbr"].astype(str).map(mapping)
    if assignments.isna().any():
        raise AssertionError("At least one encounter was not assigned to a split")
    assert_disjoint_patients(frame, assignments)
    return assignments.rename("split")


def assert_disjoint_patients(frame: pd.DataFrame, assignments: pd.Series) -> None:
    """Raise if any patient appears in multiple data populations."""
    labelled = pd.DataFrame(
        {"patient_nbr": frame["patient_nbr"].astype(str), "split": assignments.values}
    )
    overlaps = labelled.groupby("patient_nbr")["split"].nunique()
    if int((overlaps > 1).sum()) != 0:
        raise AssertionError("Patient overlap detected across data splits")


def save_split_assignments(
    frame: pd.DataFrame, assignments: pd.Series, path: Path
) -> pd.DataFrame:
    """Persist the common encounter/patient split assignment."""
    output = frame[["encounter_id", "patient_nbr"]].copy()
    output["split"] = assignments.values
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def split_summary(frame: pd.DataFrame, assignments: pd.Series) -> dict[str, object]:
    """Return counts and explicit overlap proof for artifacts."""
    labelled = frame[["patient_nbr", "target"]].copy()
    labelled["split"] = assignments.values
    counts = {}
    patients_by_split: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        subset = labelled[labelled["split"] == name]
        patients = set(subset["patient_nbr"].astype(str))
        patients_by_split[name] = patients
        counts[name] = {
            "encounters": int(len(subset)),
            "patients": int(len(patients)),
            "positive_count": int(subset["target"].sum()),
            "positive_prevalence": float(subset["target"].mean()),
        }
    overlaps = {
        "train_validation": len(patients_by_split["train"] & patients_by_split["validation"]),
        "train_test": len(patients_by_split["train"] & patients_by_split["test"]),
        "validation_test": len(
            patients_by_split["validation"] & patients_by_split["test"]
        ),
    }
    return {"counts": counts, "patient_overlap_counts": overlaps, "disjoint": not any(overlaps.values())}
