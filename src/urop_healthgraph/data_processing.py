"""Dataset loading, mapping, cleaning, and auditable summaries."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ExperimentConfig
from .feature_engineering import MEDICATION_COLUMNS, age_midpoint, icd9_category


REQUIRED_COLUMNS = {
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "diag_1",
    "diag_2",
    "diag_3",
}


def target_from_readmitted(value: object) -> int:
    """Convert the original three-level outcome to the binary research target."""
    return int(str(value).strip() == "<30")


def create_reduced_dataset(
    full_path: Path, reduced_path: Path, rows: int = 20_000, random_seed: int = 42
) -> pd.DataFrame:
    """Create the exact deterministic, target-stratified reduced dataset."""
    full = pd.read_csv(full_path, dtype=str)
    if len(full) < rows:
        raise ValueError(f"Full dataset has {len(full)} rows; cannot sample {rows}")
    y = full["readmitted"].map(target_from_readmitted)
    _, reduced = train_test_split(
        full, test_size=rows, stratify=y, random_state=random_seed
    )
    reduced_path.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(reduced_path, index=False)
    return reduced


def parse_ids_mapping(path: Path) -> dict[str, dict[str, str]]:
    """Parse the UCI mapping file, which contains three CSV sections."""
    mappings: dict[str, dict[str, str]] = {}
    current: str | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not any(cell.strip() for cell in row):
                current = None
                continue
            first = row[0].strip()
            if first in {
                "admission_type_id",
                "discharge_disposition_id",
                "admission_source_id",
            }:
                current = first
                mappings[current] = {}
                continue
            if current and len(row) >= 2:
                mappings[current][first] = row[1].strip()
    missing = {
        name
        for name in (
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
        )
        if not mappings.get(name)
    }
    if missing:
        raise ValueError(f"IDS mapping is missing sections: {sorted(missing)}")
    return mappings


def _mapped_description(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.map(mapping).fillna("Unknown / unmapped")


def load_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Load the selected raw or reduced data, recreating reduced mode if needed."""
    config.ensure_directories()
    if config.dataset == "reduced" and not config.dataset_path.exists():
        create_reduced_dataset(
            config.raw_dir / "diabetic_data.csv",
            config.dataset_path,
            rows=config.reduced_rows,
            random_seed=config.random_seed,
        )
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")
    frame = pd.read_csv(config.dataset_path, dtype=str)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return frame


def clean_dataset(
    raw: pd.DataFrame, mappings: dict[str, dict[str, str]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean records and derive readable, leakage-aware clinical fields."""
    frame = raw.copy()
    frame = frame.replace("?", np.nan)
    initial_missingness = {
        key: float(value)
        for key, value in frame.isna().mean().sort_values(ascending=False).items()
        if value > 0
    }

    for column in (
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ):
        readable = column.removesuffix("_id") + "_description"
        frame[readable] = _mapped_description(frame[column], mappings[column])

    disposition = frame["discharge_disposition_description"].str.lower()
    excluded_mask = disposition.str.contains(
        r"\b(?:expir(?:ed|ation)|death|deceased|hospice)\b",
        regex=True,
        na=False,
    )
    excluded_descriptions = sorted(
        frame.loc[excluded_mask, "discharge_disposition_description"].unique().tolist()
    )
    frame = frame.loc[~excluded_mask].copy()

    frame["target"] = frame["readmitted"].map(target_from_readmitted).astype("int8")
    frame["max_glu_serum"] = frame["max_glu_serum"].fillna("Not Measured")
    frame["A1Cresult"] = frame["A1Cresult"].fillna("Not Measured")
    frame["age_group"] = frame["age"].fillna("Unknown")
    frame["age_midpoint"] = frame["age"].map(age_midpoint)
    for diag in ("diag_1", "diag_2", "diag_3"):
        frame[f"{diag}_category"] = frame[diag].map(icd9_category)

    for medication in MEDICATION_COLUMNS:
        if medication in frame:
            values = frame[medication].fillna("No").astype(str).str.strip()
            frame[f"med_{medication}_active"] = values.isin(
                {"Up", "Down", "Steady"}
            ).astype("int8")

    removed_columns = [
        {
            "column": "weight",
            "reason": "Unusable: more than 96% of selected records are missing.",
            "missing_fraction": float(frame["weight"].isna().mean()),
        }
    ]
    if "weight" in frame:
        frame = frame.drop(columns=["weight"])

    summary: dict[str, Any] = {
        "input_rows": int(len(raw)),
        "eligible_rows": int(len(frame)),
        "excluded_end_of_life_rows": int(excluded_mask.sum()),
        "excluded_discharge_descriptions": excluded_descriptions,
        "unique_patients": int(frame["patient_nbr"].nunique()),
        "positive_count": int(frame["target"].sum()),
        "negative_count": int((frame["target"] == 0).sum()),
        "positive_prevalence": float(frame["target"].mean()),
        "initial_missingness": initial_missingness,
        "removed_columns": removed_columns,
        "source_columns": int(raw.shape[1]),
    }
    return frame.reset_index(drop=True), summary


def dataset_variant_summary(config: ExperimentConfig) -> dict[str, Any]:
    """Return lightweight full/reduced dataset counts for the dashboard."""
    result: dict[str, Any] = {}
    paths = {
        "full": config.raw_dir / "diabetic_data.csv",
        "reduced": config.processed_dir / "diabetic_data_reduced.csv",
    }
    for name, path in paths.items():
        if path.exists():
            small = pd.read_csv(
                path, usecols=["encounter_id", "patient_nbr", "readmitted"], dtype=str
            )
            target = small["readmitted"].map(target_from_readmitted)
            result[name] = {
                "rows": int(len(small)),
                "unique_patients": int(small["patient_nbr"].nunique()),
                "positive_count": int(target.sum()),
                "positive_prevalence": float(target.mean()),
            }
    return result


def distribution_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Build compact plot-ready distributions used by the dataset page."""

    def counts(column: str, limit: int = 15) -> list[dict[str, Any]]:
        series = frame[column].fillna("Unknown").astype(str).value_counts().head(limit)
        return [{"label": str(label), "count": int(count)} for label, count in series.items()]

    active_medications = {
        medication: int(frame.get(f"med_{medication}_active", pd.Series(dtype=int)).sum())
        for medication in MEDICATION_COLUMNS
    }
    active_medications = dict(
        sorted(active_medications.items(), key=lambda item: item[1], reverse=True)[:12]
    )
    return {
        "age": counts("age_group"),
        "admission_type": counts("admission_type_description"),
        "primary_diagnosis": counts("diag_1_category"),
        "readmission": [
            {"label": "Readmitted <30 days", "count": int(frame["target"].sum())},
            {"label": "Not within 30 days", "count": int((frame["target"] == 0).sum())},
        ],
        "active_medications": [
            {"label": key, "count": value} for key, value in active_medications.items()
        ],
    }


def build_data_dictionary(
    columns: list[str], mappings: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Build searchable source-column metadata and preserve all ID descriptions."""
    definitions = {
        "encounter_id": "Unique encounter identifier; metadata only and never a predictor.",
        "patient_nbr": "De-identified patient identifier; grouping metadata, never a predictor.",
        "race": "Recorded race category.",
        "gender": "Recorded gender category.",
        "age": "Ten-year age interval, converted to a numeric midpoint for modelling.",
        "weight": "Weight interval; removed after audit because more than 96% is missing.",
        "admission_type_id": "Admission type identifier, mapped through IDS_mapping.csv.",
        "discharge_disposition_id": "Discharge destination identifier, mapped and used for eligibility audit.",
        "admission_source_id": "Admission source identifier, mapped through IDS_mapping.csv.",
        "time_in_hospital": "Length of hospital stay in days.",
        "payer_code": "Payer category code; missing values become Unknown.",
        "medical_specialty": "Admitting physician specialty category; no individual doctor identifier.",
        "num_lab_procedures": "Number of laboratory procedures during the encounter.",
        "num_procedures": "Number of non-laboratory procedures during the encounter.",
        "num_medications": "Number of distinct generic medication names administered.",
        "number_outpatient": "Outpatient visits in the preceding year.",
        "number_emergency": "Emergency visits in the preceding year.",
        "number_inpatient": "Inpatient visits in the preceding year.",
        "diag_1": "Primary ICD-9 diagnosis, mapped to a broad disease category.",
        "diag_2": "Secondary ICD-9 diagnosis, mapped to a broad disease category.",
        "diag_3": "Additional ICD-9 diagnosis, mapped to a broad disease category.",
        "number_diagnoses": "Number of diagnoses recorded for the encounter.",
        "max_glu_serum": "Maximum serum glucose result or Not Measured.",
        "A1Cresult": "A1C test result or Not Measured.",
        "change": "Whether diabetes medication was changed.",
        "diabetesMed": "Whether a diabetes medication was prescribed.",
        "readmitted": "Original outcome: <30, >30, or NO; converted to the binary target.",
    }
    entries = []
    for column in columns:
        if column in MEDICATION_COLUMNS:
            description = (
                f"Status of {column}; Up, Down, or Steady becomes active and No inactive."
            )
            role = "Medication status"
        elif column in {"encounter_id", "patient_nbr"}:
            description = definitions[column]
            role = "Metadata"
        elif column == "readmitted":
            description = definitions[column]
            role = "Target source"
        else:
            description = definitions.get(column, "Source encounter attribute.")
            role = "Predictor source"
        entries.append({"column": column, "role": role, "description": description})
    return {
        "columns": entries,
        "id_mappings": mappings,
        "search_note": "Descriptions reflect implemented transformations, not clinical advice.",
    }
