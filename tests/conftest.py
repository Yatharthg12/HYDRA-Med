"""Shared test setup and compact synthetic clinical records."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from urop_healthgraph.feature_engineering import MEDICATION_COLUMNS


@pytest.fixture
def clinical_frame() -> pd.DataFrame:
    """Return graph-ready rows with every supported relation field."""
    records = []
    diagnoses = ["Diabetes", "Circulatory", "Respiratory", "Digestive"]
    for index in range(18):
        record = {
            "encounter_id": str(1000 + index),
            "patient_nbr": str(500 + index // 2),
            "target": index % 7 == 0,
            "age_group": "[60-70)",
            "diag_1_category": diagnoses[index % len(diagnoses)],
            "diag_2_category": diagnoses[(index + 1) % len(diagnoses)],
            "diag_3_category": "Other",
            "admission_type_description": "Emergency" if index % 2 else "Elective",
            "admission_source_description": "Emergency Room",
            "A1Cresult": "Not Measured" if index % 3 else ">8",
            "max_glu_serum": "Not Measured",
        }
        for medication in MEDICATION_COLUMNS:
            record[f"med_{medication}_active"] = int(
                medication == "insulin" or (medication == "metformin" and index % 2)
            )
        records.append(record)
    return pd.DataFrame(records)
