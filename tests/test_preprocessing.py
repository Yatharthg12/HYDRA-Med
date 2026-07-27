from pathlib import Path

import numpy as np
import pandas as pd

from urop_healthgraph.data_processing import (
    create_reduced_dataset,
    target_from_readmitted,
)
from urop_healthgraph.feature_engineering import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    icd9_category,
    make_preprocessor,
)


def test_target_conversion() -> None:
    assert target_from_readmitted("<30") == 1
    assert target_from_readmitted("NO") == 0
    assert target_from_readmitted(">30") == 0


def test_reduced_dataset_reproducibility() -> None:
    root = Path(__file__).resolve().parents[1]
    temporary = root / "tests" / "_generated" / "reduced.csv"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        generated = create_reduced_dataset(
            root / "data" / "raw" / "diabetic_data.csv",
            temporary,
            rows=20_000,
            random_seed=42,
        )
        existing = pd.read_csv(
            root / "data" / "processed" / "diabetic_data_reduced.csv", dtype=str
        )
        assert generated.reset_index(drop=True).equals(existing.reset_index(drop=True))
    finally:
        temporary.unlink(missing_ok=True)


def test_icd_category_mapping() -> None:
    expected = {
        "250.13": "Diabetes",
        "414": "Circulatory",
        "486": "Respiratory",
        "530": "Digestive",
        "585": "Genitourinary",
        "715": "Musculoskeletal",
        "820": "Injury",
        "174": "Neoplasm",
        "V45": "Other",
        "?": "Unknown",
    }
    assert {code: icd9_category(code) for code in expected} == expected


def test_preprocessing_is_fit_on_training_data_only() -> None:
    rows = 5
    values: dict[str, object] = {
        column: np.arange(1, rows + 1, dtype=float) for column in NUMERIC_COLUMNS
    }
    for column in CATEGORICAL_COLUMNS:
        values[column] = ["Common", "Common", "Common", "RareTrain", "HoldoutOnly"]
    frame = pd.DataFrame(values)
    frame.loc[3, NUMERIC_COLUMNS[0]] = np.nan
    frame.loc[4, NUMERIC_COLUMNS[0]] = 10_000
    preprocessor = make_preprocessor(rare_threshold=0.2, dense=True)
    preprocessor.fit(frame.iloc[:4])
    transformed = preprocessor.transform(frame.iloc[4:])
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    rare = preprocessor.named_transformers_["categorical"].named_steps["rare"]
    assert numeric_imputer.statistics_[0] == 2.0
    assert all("HoldoutOnly" not in values for values in rare.frequent_values_)
    assert transformed.shape[0] == 1
