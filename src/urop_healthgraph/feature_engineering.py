"""Clinical feature engineering and training-fitted preprocessing."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

NUMERIC_COLUMNS = [
    "age_midpoint",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
] + [f"med_{name}_active" for name in MEDICATION_COLUMNS]

CATEGORICAL_COLUMNS = [
    "race",
    "gender",
    "payer_code",
    "medical_specialty",
    "admission_type_description",
    "discharge_disposition_description",
    "admission_source_description",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
]


def age_midpoint(value: object) -> float:
    """Convert an age interval such as ``[70-80)`` to its midpoint."""
    if value is None or pd.isna(value):
        return np.nan
    numbers = re.findall(r"\d+", str(value))
    if len(numbers) != 2:
        return np.nan
    return (float(numbers[0]) + float(numbers[1])) / 2.0


def icd9_category(value: object) -> str:
    """Map ICD-9 codes to broad reproducible disease categories.

    V and E supplementary codes are retained as ``Other``. Diabetes codes
    250.xx are separated before the wider endocrine range.
    """
    if value is None or pd.isna(value):
        return "Unknown"
    text = str(value).strip().upper()
    if not text or text in {"?", "NAN", "NONE"}:
        return "Unknown"
    if text.startswith(("V", "E")):
        return "Other"
    try:
        code = float(text)
    except ValueError:
        return "Other"
    if 250 <= code < 251:
        return "Diabetes"
    integer = int(code)
    if 390 <= integer <= 459 or integer == 785:
        return "Circulatory"
    if 460 <= integer <= 519 or integer == 786:
        return "Respiratory"
    if 520 <= integer <= 579 or integer == 787:
        return "Digestive"
    if 580 <= integer <= 629 or integer == 788:
        return "Genitourinary"
    if 710 <= integer <= 739:
        return "Musculoskeletal"
    if 800 <= integer <= 999:
        return "Injury"
    if 140 <= integer <= 239:
        return "Neoplasm"
    return "Other"


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select engineered predictors while excluding patient and encounter IDs."""
    required = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Engineered feature columns missing: {sorted(missing)}")
    features = frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group categories below a frequency learned exclusively during ``fit``."""

    def __init__(self, threshold: float = 0.005):
        self.threshold = threshold

    def fit(self, X: object, y: object = None) -> "RareCategoryGrouper":
        values = np.asarray(X, dtype=object)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.n_features_in_ = values.shape[1]
        self.frequent_values_: list[set[str]] = []
        minimum = max(1, int(np.ceil(len(values) * self.threshold)))
        for index in range(values.shape[1]):
            series = pd.Series(values[:, index]).astype(str)
            counts = series.value_counts()
            self.frequent_values_.append(set(counts[counts >= minimum].index))
        return self

    def transform(self, X: object) -> np.ndarray:
        values = np.asarray(X, dtype=object)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        result = values.astype(object, copy=True)
        for index, frequent in enumerate(self.frequent_values_):
            result[:, index] = [
                str(value) if str(value) in frequent else "Other"
                for value in result[:, index]
            ]
        return result

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features, dtype=object)


def make_preprocessor(
    rare_threshold: float = 0.005, dense: bool = False
) -> ColumnTransformer:
    """Create an unfitted numeric/categorical preprocessing transformer."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("rare", RareCategoryGrouper(rare_threshold)),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_COLUMNS), ("categorical", categorical, CATEGORICAL_COLUMNS)],
        sparse_threshold=0.0 if dense else 0.3,
        verbose_feature_names_out=True,
    )


def readable_feature_name(name: str) -> str:
    """Turn a ColumnTransformer feature label into dashboard-friendly text."""
    result = name.replace("numeric__", "").replace("categorical__", "")
    result = result.replace("med_", "").replace("_active", " active")
    result = result.replace("_", " ")
    return result.strip().title()
