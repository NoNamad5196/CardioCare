from __future__ import annotations

import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "heart_disease.csv"
RAW_DATA_DIR = DATA_DIR / "raw"

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
UCI_SOURCE_FILES = (
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
)

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]
TARGET_COLUMN = "target"
FEATURE_COLUMNS = [column for column in COLUMNS if column != TARGET_COLUMN]
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [column for column in FEATURE_COLUMNS if column not in CONTINUOUS_FEATURES]

CLINICAL_RANGES = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (0, 300),
    "chol": (0, 600),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (0, 250),
    "exang": (0, 1),
    "oldpeak": (-5, 10),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_uci_sources(raw_dir: Path = RAW_DATA_DIR, overwrite: bool = False) -> list[Path]:
    """Download the four UCI processed Heart Disease files deterministically."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths: list[Path] = []
    for filename in UCI_SOURCE_FILES:
        destination = raw_dir / filename
        if destination.exists() and not overwrite:
            downloaded_paths.append(destination)
            continue

        url = f"{UCI_BASE_URL}/{filename}"
        logging.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=30) as response:
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output)
        downloaded_paths.append(destination)
    return downloaded_paths


def _read_uci_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values=["?", ""],
        skipinitialspace=True,
    )


def clean_heart_disease_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the UCI schema, binary-encode target, and remove exact duplicates."""
    cleaned = df.copy()
    cleaned = cleaned.replace("?", np.nan)
    for column in COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned[TARGET_COLUMN] = (cleaned[TARGET_COLUMN] > 0).astype(int)
    for column, (minimum, maximum) in CLINICAL_RANGES.items():
        invalid_mask = cleaned[column].notna() & ~cleaned[column].between(minimum, maximum)
        cleaned.loc[invalid_mask, column] = np.nan
    cleaned = cleaned.drop_duplicates(ignore_index=True)
    return cleaned[COLUMNS]


def prepare_dataset(data_path: Path = DATA_PATH, overwrite: bool = False) -> pd.DataFrame:
    """Create the curated CSV from UCI sources, or load it if it already exists."""
    ensure_directories()
    if data_path.exists() and not overwrite:
        return load_dataset(data_path)

    source_paths = download_uci_sources(RAW_DATA_DIR, overwrite=False)
    frames = [_read_uci_file(path) for path in source_paths]
    dataset = clean_heart_disease_frame(pd.concat(frames, ignore_index=True))
    dataset.to_csv(data_path, index=False)
    return dataset


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        return prepare_dataset(data_path)

    df = pd.read_csv(data_path, na_values=["?", ""])
    missing_columns = [column for column in COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return clean_heart_disease_frame(df[COLUMNS])


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].astype(int).copy()


def validate_feature_ranges(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate known clinical input ranges while allowing missing values for imputation."""
    ranges = ranges or CLINICAL_RANGES
    errors: list[str] = []

    for column, (minimum, maximum) in ranges.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = numeric.notna() & ~numeric.between(minimum, maximum)
        if invalid_mask.any():
            bad_values = numeric[invalid_mask].head(5).tolist()
            errors.append(
                f"{column} must be between {minimum} and {maximum}; "
                f"found examples {bad_values}"
            )

    if errors and raise_on_error:
        raise ValueError("; ".join(errors))
    return errors


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip continuous features using train-fold IQR bounds."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        array = self._as_array(X)
        self.q1_ = np.nanpercentile(array, 25, axis=0)
        self.q3_ = np.nanpercentile(array, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_bounds_ = self.q1_ - self.factor * iqr
        self.upper_bounds_ = self.q3_ + self.factor * iqr
        return self

    def transform(self, X):
        array = self._as_array(X).copy()
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self.lower_bounds_))]
        return np.asarray(input_features, dtype=object)

    @staticmethod
    def _as_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)


def build_preprocessor() -> ColumnTransformer:
    continuous_pipeline = Pipeline(
        steps=[
            ("iqr_clip", IQRClipper(factor=1.5)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, CONTINUOUS_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def iqr_outlier_summary(df: pd.DataFrame, columns: Iterable[str] = CONTINUOUS_FEATURES) -> pd.DataFrame:
    rows = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": count,
            }
        )
    return pd.DataFrame(rows)
