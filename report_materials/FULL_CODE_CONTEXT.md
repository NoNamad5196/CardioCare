# CardioCare Full Code Context

이 파일은 Claude 보고서 작성을 위해 주요 코드와 설정 파일을 하나의 Markdown으로 합친 자료입니다. 원본 파일도 ZIP의 code/ 폴더에 함께 들어 있습니다.

## README.md

````markdown
# CardioCare

CardioCare는 UCI Heart Disease 데이터를 이용해 심장병 가능성을 이진 분류하는 종단간 머신러닝 프로젝트입니다. 이 프로젝트의 목적은 심장 전문의의 판단을 보조하는 것이며, 모델이 진단이나 치료 결정을 단독으로 내리는 시스템이 아닙니다.

GitHub 저장소: https://github.com/NoNamad5196/CardioCare

## 1. 프로젝트 개요

- 문제: 환자 임상 지표를 바탕으로 심장병 여부를 예측합니다.
- 데이터: UCI Heart Disease 데이터셋의 통합 버전, 총 918행과 13개 입력 특성 사용.
- 타깃: 원래 다중 클래스인 `num` 값을 `0 = 정상`, `1 = 심장병 있음`으로 이진화했습니다.
- 최종 모델: `SVC`
- 모델 버전: `cardiocare-1.0`
- 주요 산출물: 학습 코드, 추론 코드, 모니터링 코드, unittest, Dockerfile, GitHub Actions CI, MLflow 실행 기록, 보고서 자료.

## 2. 재현 방법

Python 3.10 이상을 사용합니다. 가상환경 사용을 권장합니다.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

전체 파이프라인 실행:

```bash
python src/train.py
python src/inference.py --input data/sample_input.csv --output artifacts/predictions.csv
python src/monitor.py
python src/report_assets.py
python -m unittest
```

Docker Desktop 또는 Docker daemon이 실행 중인 상태에서:

```bash
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

## 3. 저장소 구조

```text
data/
  heart_disease.csv
  sample_input.csv
  raw/
notebooks/
  01_eda_preprocessing.ipynb
src/
  preprocessing.py
  train.py
  inference.py
  monitor.py
  report.py
  report_assets.py
tests/
  test_pipeline.py
artifacts/
models/
mlruns/
report_materials/
.github/workflows/ci.yml
Dockerfile
requirements.txt
README.md
report.pdf
```

## 4. 데이터와 전처리

데이터가 없을 경우 `src/preprocessing.py`가 UCI의 네 개 processed 파일을 내려받아 통합 데이터셋을 생성합니다.

- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

전처리는 scikit-learn `Pipeline`과 `ColumnTransformer`로 구성했습니다. 데이터 누수를 막기 위해 train/test split 이후 학습 fold 안에서만 imputer, scaler, feature selector가 fit됩니다.

전처리 방식:

- 연속형 특성: IQR 기반 clipping, median imputation, `StandardScaler`
- 범주형/코드형 특성: most-frequent imputation, one-hot encoding
- 특성 선택: `SelectFromModel(RandomForestClassifier(random_state=42))`

## 5. 학습과 실험 관리

`src/train.py`는 다음 모델을 학습하고 비교합니다.

- Logistic Regression
- SVC
- Random Forest

각 실행은 MLflow에 파라미터, 5-fold cross-validation 지표, test 지표, confusion matrix, 선택된 특성, 학습 모델 artifact를 기록합니다. 강한 후보 모델에 대해서는 `GridSearchCV` 기반 하이퍼파라미터 탐색도 수행합니다.

최종 선택 기준은 임상적 의사결정 보조 맥락을 반영해 balanced accuracy가 최고 성능에서 0.03 이내인 모델 중 recall을 우선하고, 그 다음 F1과 balanced accuracy를 봅니다. false negative는 실제 질환 가능성이 있는 환자를 놓치는 경우라서 특히 중요하게 다뤘습니다.

최종 모델 성능:

| 항목 | 값 |
| --- | ---: |
| selected run | `baseline_svc` |
| balanced accuracy | 0.8387 |
| precision | 0.8476 |
| recall | 0.8725 |
| F1 | 0.8599 |

## 6. 추론

```bash
python src/inference.py --input data/sample_input.csv --output artifacts/predictions.csv
```

입력 CSV에는 13개 feature column이 필요합니다. `target` column이 있으면 결과 파일과 로그에 함께 보존됩니다.

추론 로그에는 다음 항목이 저장됩니다.

- timestamp
- model version
- input shape
- predictions
- 실제 정답이 있는 경우 actual labels

## 7. 테스트와 CI

테스트 실행:

```bash
python -m unittest
```

`tests/test_pipeline.py`는 다음을 검증합니다.

- 예측 결과 shape가 입력 행 수와 일치하는지
- 예측 확률이 `[0, 1]` 범위에 있고 각 행의 합이 1에 가까운지
- `chol` 값이 임상적으로 허용한 범위를 벗어나면 검증 오류가 나는지
- 고정 시드에서 같은 입력이 같은 예측을 내는지

GitHub Actions 설정은 `.github/workflows/ci.yml`에 있으며, push와 pull request마다 Python 3.10 환경에서 `python -m unittest`를 실행합니다.

## 8. 모니터링과 드리프트

```bash
python src/monitor.py
```

`src/monitor.py`는 deterministic train/test split을 다시 만들고, test set 복사본에서 `chol`과 `oldpeak` 분포를 인위적으로 이동시킵니다. 이후 연속형 특성별로 `scipy.stats.ks_2samp`를 적용해 `p < 0.05`이면 drift로 표시합니다.

드리프트 결과:

| feature | drift flag |
| --- | --- |
| chol | True |
| oldpeak | True |
| age | False |
| trestbps | False |
| thalach | False |

성능 비교:

- 원본 test balanced accuracy: 0.8387
- drifted test balanced accuracy: 0.6280

## 9. Docker

`Dockerfile`은 `python:3.10-slim` 기반으로 의존성을 설치하고, 저장된 모델로 샘플 입력 추론을 실행합니다.

```bash
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

컨테이너 실행 시 `data/sample_input.csv`를 입력으로 사용하고 `artifacts/docker_predictions.csv`에 결과를 저장합니다.

## 10. 보고서 자료

보고서 작성용 표, 그림, CSV, JSON 자료는 `report_materials/`에 정리되어 있습니다.

주요 파일:

- `report_materials/CardioCare_report_materials.zip`
- `report_materials/01_eda_summary.png`
- `report_materials/02_model_comparison_table.png`
- `report_materials/03_drift_report_table.png`
- `report_materials/04_drift_performance_table.png`
- `report_materials/model_comparison.csv`
- `report_materials/final_model_metadata.json`
- `report_materials/drift_report.csv`
- `report_materials/drift_performance_comparison.csv`

## 11. Feature Store와 Model Registry 메모

Feature Store 후보로는 `thalach`가 적절합니다. 최대 심박수는 학습, 추론, 드리프트 모니터링에서 반복적으로 사용되므로 schema, freshness, validation을 명시적으로 관리할 가치가 있습니다.

Model Registry에 기록해야 할 주요 메타데이터는 `selected_features`입니다. 어떤 변환 특성이 최종적으로 살아남았는지 남겨야 모델 버전 간 비교와 감사 가능성이 좋아집니다.

## 12. 서빙과 재학습 전략

이 프로젝트에서는 on-device serving보다 Model-as-a-Service가 더 적합하다고 판단했습니다. 서버 기반 서빙은 모델 업데이트, rollback, 감사 로그, 모니터링을 중앙에서 관리하기 쉽습니다. 단, 실제 의료 환경에서는 PHI 보호를 위해 암호화된 전송, 엄격한 접근 제어, 최소 보존 정책이 필요합니다.

재학습은 단순히 drift가 한 번 감지되었다는 이유만으로 자동 수행하지 않습니다. drift 감지, 성능 저하, 사람의 검토를 거친 새로운 label 확보가 함께 확인될 때 재학습 후보로 올리고, 모델 promotion 전에는 Human-in-the-loop 검토가 필요합니다.

## 13. 한계와 윤리

CardioCare는 오래된 공개 데이터셋을 기반으로 한 교육용 prototype입니다. 실제 임상 적용에는 외부 검증, calibration, 공정성 분석, 개인정보 보호 설계, 의료진 검토 절차가 추가로 필요합니다. 모델은 위험 신호를 알릴 수는 있지만, 진단과 치료 결정을 대신해서는 안 됩니다.
````

## requirements.txt

````text
numpy==2.2.6
pandas==2.3.3
scipy==1.15.3
scikit-learn==1.7.2
mlflow==3.3.2
matplotlib==3.10.8
joblib==1.5.2
````

## Dockerfile

````dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY src ./src
COPY models ./models
COPY artifacts ./artifacts
COPY README.md ./README.md

RUN mkdir -p logs artifacts

CMD ["python", "src/inference.py", "--input", "data/sample_input.csv", "--output", "artifacts/docker_predictions.csv"]
````

## .github\workflows\ci.yml

````yaml
name: CardioCare CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run unit tests
        run: python -m unittest
````

## src\__init__.py

````python
"""CardioCare machine learning package."""
````

## src\preprocessing.py

````python
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
````

## src\train.py

````python
from __future__ import annotations

import json
import hashlib
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cardiocare")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    CONTINUOUS_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    build_preprocessor,
    load_dataset,
    prepare_dataset,
    save_json,
    split_features_target,
    validate_feature_ranges,
)


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
SAMPLE_INPUT_PATH = PROJECT_ROOT / "data" / "sample_input.csv"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"
FINAL_METADATA_PATH = MODELS_DIR / "final_model_metadata.json"
EXPERIMENT_NAME = "CardioCare Heart Disease Prediction"


def ensure_output_dirs() -> None:
    for path in (ARTIFACTS_DIR, MODELS_DIR, MLRUNS_DIR, SAMPLE_INPUT_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


def configure_mlflow_tracking() -> str:
    """Use a local MLflow experiment whose artifact URI matches this checkout."""
    tracking_uri = MLRUNS_DIR.resolve().as_uri()
    expected_artifact_prefix = f"{tracking_uri.rstrip('/')}/"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    candidate_names = [
        EXPERIMENT_NAME,
        f"{EXPERIMENT_NAME} (local)",
    ]
    path_fingerprint = hashlib.sha1(str(PROJECT_ROOT.resolve()).encode("utf-8")).hexdigest()[:8]
    candidate_names.append(f"{EXPERIMENT_NAME} ({path_fingerprint})")

    for name in candidate_names:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            mlflow.set_experiment(name)
            return name
        artifact_location = str(experiment.artifact_location).replace("\\", "/")
        if artifact_location.startswith(expected_artifact_prefix):
            mlflow.set_experiment(name)
            return name

    fallback_name = f"{EXPERIMENT_NAME} ({path_fingerprint}-run)"
    mlflow.set_experiment(fallback_name)
    return fallback_name


def build_pipeline(classifier) -> Pipeline:
    selector_estimator = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("feature_selection", SelectFromModel(selector_estimator, threshold="median")),
            ("classifier", classifier),
        ]
    )


def model_definitions() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "svc": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def tuning_grid(model_name: str) -> dict[str, list[object]]:
    grids = {
        "logistic_regression": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
        "svc": {
            "classifier__C": [0.5, 1.0, 2.0],
            "classifier__gamma": ["scale", 0.05],
        },
        "random_forest": {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_leaf": [1, 3],
            "classifier__max_features": ["sqrt", None],
        },
    }
    return grids[model_name]


def metric_scores(y_true, y_pred) -> dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, run_name: str) -> tuple[Path, Path]:
    matrix = confusion_matrix(y_true, y_pred)
    csv_path = ARTIFACTS_DIR / f"{run_name}_confusion_matrix.csv"
    png_path = ARTIFACTS_DIR / f"{run_name}_confusion_matrix.png"
    pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{run_name} confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, png_path


def selected_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    selector = pipeline.named_steps["feature_selection"]
    feature_names = preprocessor.get_feature_names_out()
    return feature_names[selector.get_support()].tolist()


def log_model_run(
    run_name: str,
    model_family: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
    extra_params: dict[str, object] | None = None,
) -> dict[str, object]:
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    fitted_pipeline = clone(pipeline)
    fitted_pipeline.fit(X_train, y_train)
    y_pred = fitted_pipeline.predict(X_test)
    test_metrics = metric_scores(y_test, y_pred)
    cm_csv, cm_png = save_confusion_matrix(y_test, y_pred, run_name)
    features = selected_feature_names(fitted_pipeline)
    feature_path = ARTIFACTS_DIR / f"{run_name}_selected_features.txt"
    feature_path.write_text("\n".join(features) + "\n", encoding="utf-8")

    row: dict[str, object] = {
        "run_name": run_name,
        "model_family": model_family,
        "selected_feature_count": len(features),
        "model": fitted_pipeline,
        "confusion_matrix_csv": str(cm_csv),
        "confusion_matrix_png": str(cm_png),
        "selected_features_path": str(feature_path),
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_family", model_family)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("feature_selector", "SelectFromModel(RandomForestClassifier)")
        mlflow.log_param("selected_feature_count", len(features))
        if extra_params:
            for key, value in extra_params.items():
                mlflow.log_param(key, value)

        classifier_params = fitted_pipeline.named_steps["classifier"].get_params()
        for key, value in classifier_params.items():
            if key in {"random_state", "class_weight", "C", "gamma", "kernel", "n_estimators", "max_depth", "min_samples_leaf", "max_features"}:
                mlflow.log_param(f"classifier__{key}", value)

        for metric_name in scoring:
            values = cv_results[f"test_{metric_name}"]
            mean_value = float(np.mean(values))
            std_value = float(np.std(values))
            mlflow.log_metric(f"cv_{metric_name}_mean", mean_value)
            mlflow.log_metric(f"cv_{metric_name}_std", std_value)
            row[f"cv_{metric_name}_mean"] = mean_value
            row[f"cv_{metric_name}_std"] = std_value

        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", float(value))
            row[f"test_{metric_name}"] = float(value)

        mlflow.log_artifact(cm_csv)
        mlflow.log_artifact(cm_png)
        mlflow.log_artifact(feature_path)
        mlflow.sklearn.log_model(
            fitted_pipeline,
            name="model",
            input_example=X_train.head(3).astype(float),
        )

    return row


def choose_final_model(results: list[dict[str, object]]) -> dict[str, object]:
    best_balanced_accuracy = max(float(row["test_balanced_accuracy"]) for row in results)
    eligible = [
        row
        for row in results
        if float(row["test_balanced_accuracy"]) >= best_balanced_accuracy - 0.03
    ]
    return sorted(
        eligible,
        key=lambda row: (
            float(row["test_recall"]),
            float(row["test_f1"]),
            float(row["test_balanced_accuracy"]),
        ),
        reverse=True,
    )[0]


def run_training() -> dict[str, object]:
    ensure_output_dirs()
    dataset = prepare_dataset(DATA_PATH)
    X, y = split_features_target(dataset)
    validate_feature_ranges(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    configure_mlflow_tracking()

    results: list[dict[str, object]] = []
    pipelines = {
        name: build_pipeline(classifier)
        for name, classifier in model_definitions().items()
    }

    for model_name, pipeline in pipelines.items():
        print(f"\n=== Training baseline: {model_name} ===")
        result = log_model_run(
            run_name=f"baseline_{model_name}",
            model_family=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            extra_params={"run_type": "baseline"},
        )
        results.append(result)
        print(
            json.dumps(
                {key: result[key] for key in result if key.startswith("test_")},
                indent=2,
            )
        )
        print(pd.read_csv(result["confusion_matrix_csv"], index_col=0))

    strongest_baseline = max(results, key=lambda row: float(row["cv_balanced_accuracy_mean"]))
    tuned_name = str(strongest_baseline["model_family"])
    print(f"\n=== Hyperparameter tuning: {tuned_name} ===")
    grid = GridSearchCV(
        estimator=pipelines[tuned_name],
        param_grid=tuning_grid(tuned_name),
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    tuned_pipeline = grid.best_estimator_
    tuned_result = log_model_run(
        run_name=f"tuned_{tuned_name}",
        model_family=tuned_name,
        pipeline=tuned_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
        extra_params={
            "run_type": "tuned",
            "best_cv_balanced_accuracy": float(grid.best_score_),
            "best_params": json.dumps(grid.best_params_, sort_keys=True),
        },
    )
    results.append(tuned_result)

    comparison_columns = [
        "run_name",
        "model_family",
        "selected_feature_count",
        "cv_balanced_accuracy_mean",
        "cv_precision_mean",
        "cv_recall_mean",
        "cv_f1_mean",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    comparison = pd.DataFrame([{key: row[key] for key in comparison_columns} for row in results])
    comparison_path = ARTIFACTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print("\n=== Model comparison ===")
    print(comparison.to_string(index=False))

    final_row = choose_final_model(results)
    final_model = final_row["model"]
    joblib.dump(final_model, FINAL_MODEL_PATH)

    SAMPLE_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample = X_test.head(8).copy()
    sample[TARGET_COLUMN] = y_test.head(8).to_numpy()
    sample.to_csv(SAMPLE_INPUT_PATH, index=False)

    metadata = {
        "model_version": "cardiocare-1.0",
        "selected_run": final_row["run_name"],
        "model_family": final_row["model_family"],
        "random_state": RANDOM_STATE,
        "data_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "dataset_rows": int(dataset.shape[0]),
        "dataset_columns": int(dataset.shape[1]),
        "feature_columns": FEATURE_COLUMNS,
        "continuous_features": CONTINUOUS_FEATURES,
        "selected_features": selected_feature_names(final_model),
        "metrics": {
            key: float(final_row[key])
            for key in final_row
            if key.startswith("test_") or key.startswith("cv_")
        },
        "selection_rationale": (
            "The final model is selected from models within 0.03 balanced-accuracy "
            "of the best test result, prioritizing recall to reduce false negatives "
            "in a clinical decision-support setting, then F1 and balanced accuracy."
        ),
    }
    save_json(metadata, FINAL_METADATA_PATH)

    print("\n=== Final model ===")
    print(json.dumps({k: v for k, v in metadata.items() if k != "selected_features"}, indent=2))
    print("\nClinical rationale:")
    print(
        "CardioCare is a decision-support model, not an autonomous diagnosis system. "
        "False negatives are clinically costly because a patient with possible disease "
        "could be incorrectly treated as low risk. The selected model therefore favors "
        "recall among models with comparable balanced accuracy, while still reporting "
        "precision and F1 so clinicians can understand false-positive burden."
    )

    return metadata


if __name__ == "__main__":
    run_training()
````

## src\inference.py

````python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, validate_feature_ranges  # noqa: E402


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "final_model_metadata.json"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sample_input.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "predictions.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "inference.log"


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {"model_version": "unknown"}
    return json.loads(path.read_text(encoding="utf-8"))


def run_inference(
    input_path: Path = DEFAULT_INPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    configure_logging()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python src/train.py` first."
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    model = joblib.load(model_path)
    metadata = load_metadata(metadata_path)
    input_df = pd.read_csv(input_path)
    actual = input_df[TARGET_COLUMN].copy() if TARGET_COLUMN in input_df.columns else None
    features = input_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    features = features[FEATURE_COLUMNS]
    validate_feature_ranges(features)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    output = features.copy()
    output["prediction"] = predictions
    output["probability_0"] = probabilities[:, 0]
    output["probability_1"] = probabilities[:, 1]
    if actual is not None:
        output["actual"] = actual.to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    logging.info(
        "model_version=%s input_shape=%s predictions=%s actual=%s",
        metadata.get("model_version", "unknown"),
        tuple(features.shape),
        predictions.tolist(),
        actual.tolist() if actual is not None else None,
    )
    print(output[["prediction", "probability_0", "probability_1"]].to_string(index=False))
    print(f"Saved predictions to {output_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CardioCare batch inference.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.input, args.model, args.metadata, args.output)
````

## src\monitor.py

````python
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cardiocare")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    CONTINUOUS_FEATURES,
    DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    load_dataset,
    split_features_target,
)


MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "final_model_metadata.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_PATH = PROJECT_ROOT / "logs" / "monitor.log"


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {"model_version": "unknown"}


def drift_test(X_train: pd.DataFrame, X_candidate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in CONTINUOUS_FEATURES:
        reference = pd.to_numeric(X_train[feature], errors="coerce").dropna()
        candidate = pd.to_numeric(X_candidate[feature], errors="coerce").dropna()
        statistic, p_value = ks_2samp(reference, candidate)
        rows.append(
            {
                "feature": feature,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_flag": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)


def make_drifted_copy(X_test: pd.DataFrame, shift: float = 30.0) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    drifted = X_test.copy()
    chol = pd.to_numeric(drifted["chol"], errors="coerce")
    noise = rng.normal(loc=0.0, scale=20.0, size=len(drifted))
    drifted["chol"] = (chol * 1.15 + shift + noise).clip(lower=0, upper=600)
    oldpeak = pd.to_numeric(drifted["oldpeak"], errors="coerce")
    oldpeak_noise = rng.normal(loc=0.0, scale=0.2, size=len(drifted))
    drifted["oldpeak"] = (oldpeak + shift / 10.0 + oldpeak_noise).clip(lower=-5, upper=10)
    return drifted


def metric_timeseries(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    rows = []
    for idx, shift in enumerate([0, 15, 30, 45, 60], start=1):
        candidate = X_test.copy() if shift == 0 else make_drifted_copy(X_test, shift=float(shift))
        predictions = model.predict(candidate)
        rows.append(
            {
                "timestamp": pd.Timestamp("2026-05-01") + pd.Timedelta(days=7 * idx),
                "chol_shift": shift,
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            }
        )
    return pd.DataFrame(rows)


def run_monitoring() -> dict[str, float]:
    configure_logging()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing final model. Run `python src/train.py` before monitoring.")

    metadata = load_metadata()
    model = joblib.load(MODEL_PATH)
    dataset = load_dataset(DATA_PATH)
    X, y = split_features_target(dataset)
    X_train, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    original_predictions = model.predict(X_test)
    original_balanced_accuracy = balanced_accuracy_score(y_test, original_predictions)
    drifted_X = make_drifted_copy(X_test)
    drifted_predictions = model.predict(drifted_X)
    drifted_balanced_accuracy = balanced_accuracy_score(y_test, drifted_predictions)

    drift_report = drift_test(X_train, drifted_X)
    drift_report_path = ARTIFACTS_DIR / "drift_report.csv"
    drift_report.to_csv(drift_report_path, index=False)

    performance = pd.DataFrame(
        [
            {"dataset": "original_test", "balanced_accuracy": original_balanced_accuracy},
            {"dataset": "drifted_test", "balanced_accuracy": drifted_balanced_accuracy},
        ]
    )
    performance_path = ARTIFACTS_DIR / "drift_performance_comparison.csv"
    performance.to_csv(performance_path, index=False)

    timeseries = metric_timeseries(model, X_test, y_test)
    timeseries_path = ARTIFACTS_DIR / "performance_timeseries.csv"
    timeseries.to_csv(timeseries_path, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(timeseries["timestamp"], timeseries["balanced_accuracy"], marker="o", linewidth=2)
    ax.set_title("Balanced accuracy over synthetic drift windows")
    ax.set_xlabel("Synthetic timestamp")
    ax.set_ylabel("Balanced accuracy")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    plot_path = ARTIFACTS_DIR / "drift_performance_timeseries.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    logging.info(
        "model_version=%s input_shape=%s original_predictions=%s drifted_predictions=%s actual=%s",
        metadata.get("model_version", "unknown"),
        tuple(X_test.shape),
        original_predictions.tolist(),
        drifted_predictions.tolist(),
        y_test.tolist(),
    )

    summary = {
        "original_balanced_accuracy": float(original_balanced_accuracy),
        "drifted_balanced_accuracy": float(drifted_balanced_accuracy),
        "delta": float(drifted_balanced_accuracy - original_balanced_accuracy),
        "drifted_features": drift_report.loc[drift_report["drift_flag"], "feature"].tolist(),
    }
    summary_path = ARTIFACTS_DIR / "monitor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== KS drift report ===")
    print(drift_report.to_string(index=False))
    print("\n=== Performance comparison ===")
    print(performance.to_string(index=False))
    print(f"\nSaved monitor artifacts to {ARTIFACTS_DIR}")
    return summary


if __name__ == "__main__":
    run_monitoring()
````

## src\report_assets.py

````python
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cardiocare")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "report_materials"
ZIP_PATH = OUTPUT_DIR / "CardioCare_report_materials.zip"
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def configure_fonts() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def clean_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for child in OUTPUT_DIR.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def save_table_png(df: pd.DataFrame, title: str, path: Path, font_size: float = 9.0) -> None:
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.3f}")

    fig_height = max(2.6, 0.45 * len(display) + 1.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#18314f", pad=14)
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e6eef8")
            cell.set_text_props(weight="bold", color="#18314f")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_eda_png() -> None:
    df = pd.read_csv(DATA_PATH, na_values=["?", ""])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.patch.set_facecolor("white")
    fig.suptitle("CardioCare EDA 요약", x=0.04, y=1.02, ha="left", fontsize=18, fontweight="bold", color="#18314f")

    target_counts = df["target"].value_counts(normalize=True).sort_index()
    axes[0].bar(["0: 정상", "1: 심장병"], target_counts.values, color=["#4c78a8", "#f58518"])
    axes[0].set_title("타깃 클래스 비율")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("비율")
    for idx, value in enumerate(target_counts.values):
        axes[0].text(idx, value + 0.025, f"{value:.1%}", ha="center", fontsize=10)

    missing_rates = df.isna().mean().sort_values(ascending=False)
    missing_rates = missing_rates[missing_rates > 0].head(6)
    axes[1].barh(missing_rates.index[::-1], missing_rates.values[::-1], color="#72b7b2")
    axes[1].set_title("결측률 상위 변수")
    axes[1].set_xlabel("결측률")
    axes[1].set_xlim(0, max(0.55, float(missing_rates.max()) * 1.2))
    for y_pos, value in enumerate(missing_rates.values[::-1]):
        axes[1].text(value + 0.01, y_pos, f"{value:.1%}", va="center", fontsize=10)

    outlier_counts = []
    for column in CONTINUOUS_FEATURES:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts.append(int(((series < lower) | (series > upper)).sum()))
    axes[2].bar(CONTINUOUS_FEATURES, outlier_counts, color="#54a24b")
    axes[2].set_title("IQR 이상치 개수")
    axes[2].tick_params(axis="x", rotation=30)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_eda_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_selected_features_file() -> None:
    metadata_path = MODELS_DIR / "final_model_metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata.get("selected_features", [])
    rows = ["# 최종 모델 선택 특성", ""]
    rows.extend(f"- {feature}" for feature in features)
    (OUTPUT_DIR / "selected_features.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def copy_existing_assets() -> None:
    copies = [
        ARTIFACTS_DIR / "model_comparison.csv",
        ARTIFACTS_DIR / "drift_report.csv",
        ARTIFACTS_DIR / "drift_performance_comparison.csv",
        ARTIFACTS_DIR / "performance_timeseries.csv",
        ARTIFACTS_DIR / "drift_performance_timeseries.png",
        ARTIFACTS_DIR / "predictions.csv",
        ARTIFACTS_DIR / "monitor_summary.json",
        MODELS_DIR / "final_model_metadata.json",
    ]
    metadata_path = MODELS_DIR / "final_model_metadata.json"
    if metadata_path.exists():
        selected_run = json.loads(metadata_path.read_text(encoding="utf-8")).get("selected_run")
        if selected_run:
            copies.extend(
                [
                    ARTIFACTS_DIR / f"{selected_run}_confusion_matrix.csv",
                    ARTIFACTS_DIR / f"{selected_run}_confusion_matrix.png",
                    ARTIFACTS_DIR / f"{selected_run}_selected_features.txt",
                ]
            )

    for source in copies:
        if source.exists():
            shutil.copy2(source, OUTPUT_DIR / source.name)


def make_table_images() -> None:
    comparison = pd.read_csv(ARTIFACTS_DIR / "model_comparison.csv")
    comparison_display = comparison.rename(
        columns={
            "run_name": "run",
            "model_family": "model",
            "selected_feature_count": "features",
            "cv_balanced_accuracy_mean": "cv_bAcc",
            "cv_precision_mean": "cv_prec",
            "cv_recall_mean": "cv_recall",
            "cv_f1_mean": "cv_F1",
            "test_balanced_accuracy": "test_bAcc",
            "test_precision": "test_prec",
            "test_recall": "test_recall",
            "test_f1": "test_F1",
        }
    )
    comparison_display["run"] = comparison_display["run"].replace(
        {
            "baseline_logistic_regression": "base_logreg",
            "baseline_random_forest": "base_rf",
            "baseline_svc": "base_svc",
            "tuned_svc": "tuned_svc",
        }
    )
    comparison_display["model"] = comparison_display["model"].replace(
        {
            "logistic_regression": "logreg",
            "random_forest": "rf",
        }
    )
    save_table_png(comparison_display, "MLflow 모델 비교표", OUTPUT_DIR / "02_model_comparison_table.png", font_size=8.5)

    drift = pd.read_csv(ARTIFACTS_DIR / "drift_report.csv")
    save_table_png(drift, "KS 검정 드리프트 결과", OUTPUT_DIR / "03_drift_report_table.png", font_size=9.5)

    performance = pd.read_csv(ARTIFACTS_DIR / "drift_performance_comparison.csv")
    save_table_png(performance, "원본 vs. 드리프트 성능 비교", OUTPUT_DIR / "04_drift_performance_table.png", font_size=10.0)


def write_readme() -> None:
    content = """# CardioCare 보고서 작성용 자료

이 zip은 report.pdf를 새로 작성할 때 바로 사용할 수 있는 그림, 표, 원본 수치 파일을 모은 것입니다.

## PNG 그림

- `01_eda_summary.png`: 타깃 분포, 결측률, IQR 이상치 요약
- `02_model_comparison_table.png`: MLflow 모델 비교표
- `03_drift_report_table.png`: KS 검정 기반 drift report
- `04_drift_performance_table.png`: 원본/드리프트 balanced accuracy 비교
- `drift_performance_timeseries.png`: synthetic drift window별 balanced accuracy 그래프
- `baseline_svc_confusion_matrix.png`: 최종 모델 confusion matrix

## 원본 데이터 파일

- `model_comparison.csv`: 모델별 CV/test metric
- `drift_report.csv`: feature별 KS statistic, p-value, drift flag
- `drift_performance_comparison.csv`: original_test vs drifted_test 성능
- `performance_timeseries.csv`: synthetic timestamp별 성능
- `predictions.csv`: sample input inference 결과
- `final_model_metadata.json`: 최종 모델, 선택 특성, metric metadata
- `monitor_summary.json`: monitoring 요약
- `selected_features.md`: 보고서에 붙이기 쉬운 최종 선택 특성 목록

보고서 문장 핵심: CardioCare는 "알리되, 결정하지 않는다"는 의사결정 보조 시스템이며, 최종 모델 선택에서는 false negative 위험 때문에 recall을 중요하게 보았습니다.
"""
    (OUTPUT_DIR / "README_report_materials.md").write_text(content, encoding="utf-8")


def create_zip() -> Path:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.iterdir()):
            if path == ZIP_PATH:
                continue
            archive.write(path, arcname=path.name)
    return ZIP_PATH


def main() -> None:
    configure_fonts()
    clean_output_dir()
    make_eda_png()
    make_table_images()
    make_selected_features_file()
    copy_existing_assets()
    write_readme()
    zip_path = create_zip()
    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
````

## src\report.py

````python
from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cardiocare")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORT_PATH = PROJECT_ROOT / "report.pdf"
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
FOOTER = "CardioCare 기계학습 기말 프로젝트 | UCI Heart Disease | 알리되, 결정하지 않는다"
KOREAN_FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")


def configure_fonts() -> None:
    if KOREAN_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(KOREAN_FONT_PATH))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def wrap_lines(text: str, width: int = 92) -> str:
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        collapsed = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if collapsed:
            paragraphs.append(collapsed)
    return "\n\n".join(
        textwrap.fill(paragraph, width=width, break_on_hyphens=False)
        for paragraph in paragraphs
    )


def format_metric(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def add_text_page(pdf: PdfPages, title: str, body: str, footer: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=18, fontweight="bold", color="#18314f")
    fig.text(
        0.08,
        0.89,
        wrap_lines(body, width=62),
        fontsize=10.5,
        va="top",
        linespacing=1.45,
        color="#1f2933",
    )
    fig.text(0.08, 0.045, footer, fontsize=8.5, color="#667085")
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
    note: str,
    footer: str,
    font_size: float = 8.5,
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color="#18314f", pad=18)
    display_df = dataframe.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.3f}")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e6eef8")
            cell.set_text_props(weight="bold", color="#18314f")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
    fig.text(0.06, 0.08, wrap_lines(note, width=108), fontsize=9.5, color="#344054")
    fig.text(0.06, 0.035, footer, fontsize=8.5, color="#667085")
    fig.tight_layout(rect=[0.04, 0.12, 0.96, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def add_eda_page(pdf: PdfPages, dataframe: pd.DataFrame, footer: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("전처리 결정을 위한 EDA 요약", x=0.06, y=0.94, ha="left", fontsize=17, fontweight="bold", color="#18314f")

    target_counts = dataframe["target"].value_counts(normalize=True).sort_index()
    axes[0].bar(["0: 정상", "1: 심장병"], target_counts.values, color=["#4c78a8", "#f58518"])
    axes[0].set_title("타깃 클래스 비율")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("비율")
    for idx, value in enumerate(target_counts.values):
        axes[0].text(idx, value + 0.025, f"{value:.1%}", ha="center", fontsize=9)

    missing_rates = dataframe.isna().mean().sort_values(ascending=False)
    missing_rates = missing_rates[missing_rates > 0].head(6)
    axes[1].barh(missing_rates.index[::-1], missing_rates.values[::-1], color="#72b7b2")
    axes[1].set_title("결측률 상위 변수")
    axes[1].set_xlabel("결측률")
    axes[1].set_xlim(0, max(0.55, float(missing_rates.max()) * 1.2 if len(missing_rates) else 0.55))
    for y_pos, value in enumerate(missing_rates.values[::-1]):
        axes[1].text(value + 0.01, y_pos, f"{value:.1%}", va="center", fontsize=9)

    outlier_rows = []
    for column in CONTINUOUS_FEATURES:
        series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_rows.append(((series < lower) | (series > upper)).sum())
    axes[2].bar(CONTINUOUS_FEATURES, outlier_rows, color="#54a24b")
    axes[2].set_title("IQR 이상치 개수")
    axes[2].tick_params(axis="x", rotation=35)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    note = (
        "그림 1. 통합 데이터는 약한 클래스 불균형, 수집 기관 차이에 따른 결측, "
        "연속형 변수의 이상치를 함께 보인다. 이 결과를 근거로 stratified split, "
        "balanced accuracy/recall 중심 평가, 대치, train-fold IQR clipping, scaling을 적용했다."
    )
    fig.text(0.06, 0.10, wrap_lines(note, width=105), fontsize=9.5, color="#344054")
    fig.text(0.06, 0.035, footer, fontsize=8.5, color="#667085")
    fig.tight_layout(rect=[0.04, 0.16, 0.96, 0.88])
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, note: str, footer: str) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.text(0.08, 0.94, title, fontsize=18, fontweight="bold", color="#18314f")
    if image_path.exists():
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_position([0.08, 0.25, 0.84, 0.56])
    else:
        fig.text(0.08, 0.55, f"Missing image: {image_path}", fontsize=11, color="#b42318")
    fig.text(0.08, 0.17, wrap_lines(note, width=62), fontsize=10, color="#344054")
    fig.text(0.08, 0.045, footer, fontsize=8.5, color="#667085")
    pdf.savefig(fig)
    plt.close(fig)


def add_final_model_page(
    pdf: PdfPages,
    final_summary: dict[str, str],
    image_path: Path,
    confusion_counts: dict[str, int] | None,
    footer: str,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, "최종 모델 선택 근거", fontsize=18, fontweight="bold", color="#18314f")

    metrics = "\n".join(
        [
            f"선택된 run: {final_summary['selected_run']}",
            f"모델 계열: {final_summary['model_family']}",
            f"테스트 balanced accuracy: {final_summary['test_balanced_accuracy']}",
            f"테스트 recall: {final_summary['test_recall']}",
            f"테스트 precision: {final_summary['test_precision']}",
            f"테스트 F1: {final_summary['test_f1']}",
        ]
    )
    fig.text(0.08, 0.885, metrics, fontsize=10.3, va="top", linespacing=1.42, color="#1f2933")

    if image_path.exists():
        ax = fig.add_axes([0.49, 0.57, 0.40, 0.27])
        ax.axis("off")
        ax.imshow(plt.imread(image_path))
    else:
        fig.text(0.50, 0.72, f"Missing image: {image_path.name}", fontsize=10, color="#b42318")

    count_sentence = "이 문제에서는 단일 accuracy보다 false negative 개수를 직접 확인하는 것이 더 중요하므로 confusion matrix를 함께 제시했다."
    if confusion_counts:
        count_sentence = (
            f"Confusion matrix 기준으로 실제 심장병 test record 중 {confusion_counts['fn']}건은 놓쳤고, "
            f"{confusion_counts['tp']}건은 올바르게 위험군으로 표시했다."
        )
    body = f"""
    최종 모델은 먼저 최고 test balanced accuracy와 0.03 이내인 후보를 남기고, 그 안에서 recall,
    F1, balanced accuracy 순으로 고르는 규칙을 따랐다. 심장병 가능성을 놓치는 false negative가
    추후 진료 지연으로 이어질 수 있기 때문에 recall을 우선순위에 둔 것이다.

    {count_sentence} 동시에 precision도 보고해 false positive로 인한 추가 검토 부담을 숨기지
    않았다. 이 모델은 교육용 의사결정 보조 도구이며, 점수는 차트를 다시 볼 이유이지 의사의
    판단을 대체할 이유가 아니다.

    그림 2. 최종 모델의 hold-out test confusion matrix.
    """
    fig.text(0.08, 0.50, wrap_lines(body, width=62), fontsize=10.2, va="top", linespacing=1.42, color="#1f2933")
    fig.text(0.08, 0.045, footer, fontsize=8.5, color="#667085")
    pdf.savefig(fig)
    plt.close(fig)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({"status": [f"Missing artifact: {path.name}. Run train.py and monitor.py."]})


def load_report_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, na_values=["?", ""])
    return pd.DataFrame(
        {
            "target": [0, 1],
            "age": [None, None],
            "trestbps": [None, None],
            "chol": [None, None],
            "thalach": [None, None],
            "oldpeak": [None, None],
        }
    )


def model_comparison_for_report(comparison: pd.DataFrame) -> pd.DataFrame:
    if "status" in comparison.columns:
        return comparison
    columns = {
        "run_name": "run",
        "model_family": "계열",
        "selected_feature_count": "특성수",
        "cv_balanced_accuracy_mean": "CV bAcc",
        "cv_precision_mean": "CV prec",
        "cv_recall_mean": "CV recall",
        "cv_f1_mean": "CV F1",
        "test_balanced_accuracy": "test bAcc",
        "test_precision": "test prec",
        "test_recall": "test recall",
        "test_f1": "test F1",
    }
    display = comparison[list(columns)].rename(columns=columns)
    display["run"] = display["run"].replace(
        {
            "baseline_logistic_regression": "base_logreg",
            "baseline_random_forest": "base_rf",
            "baseline_svc": "base_svc",
            "tuned_svc": "tuned_svc",
        }
    )
    display["계열"] = display["계열"].replace(
        {
            "logistic_regression": "logreg",
            "random_forest": "rf",
        }
    )
    return display


def final_model_summary(metadata: dict, comparison: pd.DataFrame) -> dict[str, str]:
    selected_run = metadata.get("selected_run", "baseline_random_forest")
    row = comparison.loc[comparison.get("run_name", pd.Series(dtype=str)) == selected_run]
    if row.empty:
        return {
            "selected_run": str(selected_run),
            "model_family": str(metadata.get("model_family", "unknown")),
            "test_balanced_accuracy": format_metric(metadata.get("metrics", {}).get("test_balanced_accuracy")),
            "test_recall": format_metric(metadata.get("metrics", {}).get("test_recall")),
            "test_precision": format_metric(metadata.get("metrics", {}).get("test_precision")),
            "test_f1": format_metric(metadata.get("metrics", {}).get("test_f1")),
        }

    row = row.iloc[0]
    return {
        "selected_run": str(row["run_name"]),
        "model_family": str(row["model_family"]),
        "test_balanced_accuracy": format_metric(row["test_balanced_accuracy"]),
        "test_recall": format_metric(row["test_recall"]),
        "test_precision": format_metric(row["test_precision"]),
        "test_f1": format_metric(row["test_f1"]),
    }


def confusion_counts_for_run(run_name: str) -> dict[str, int] | None:
    path = ARTIFACTS_DIR / f"{run_name}_confusion_matrix.csv"
    if not path.exists():
        return None
    matrix = pd.read_csv(path, index_col=0)
    return {
        "tn": int(matrix.loc["actual_0", "predicted_0"]),
        "fp": int(matrix.loc["actual_0", "predicted_1"]),
        "fn": int(matrix.loc["actual_1", "predicted_0"]),
        "tp": int(matrix.loc["actual_1", "predicted_1"]),
    }


def generate_report() -> Path:
    configure_fonts()
    metadata_path = PROJECT_ROOT / "models" / "final_model_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    comparison = read_optional_csv(ARTIFACTS_DIR / "model_comparison.csv")
    drift = read_optional_csv(ARTIFACTS_DIR / "drift_report.csv")
    performance = read_optional_csv(ARTIFACTS_DIR / "drift_performance_comparison.csv")
    report_dataset = load_report_dataset()
    final_summary = final_model_summary(metadata, comparison)
    final_run = final_summary["selected_run"]
    final_confusion_counts = confusion_counts_for_run(final_run)

    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(
            pdf,
            "CardioCare: 심장병 예측 종단간 ML 시스템",
            """
            CardioCare는 심장병 위험 신호를 추정하는 교육용 종단간 머신러닝 시스템이다. 이
            프로젝트의 기본 원칙은 "알리되, 결정하지 않는다"이다. 모델은 환자 record가
            심장병 위험과 얼마나 비슷한지 알려 주지만, 진단이나 치료 결정을 자동으로 내리는
            시스템이 아니다. 결과는 심장 전문의가 차트를 검토할 때 참고하는 보조 정보로만
            사용되어야 한다.

            데이터는 UCI Heart Disease의 네 개 processed 파일(Cleveland, Hungarian,
            Switzerland, VA)을 통합해 사용했다. 원래 target은 여러 질병 정도를 나타내는
            다중 클래스이지만, 이 과제에서는 target=0을 정상, target=1을 심장병 관측 기록으로
            이진화했다. repository에는 정리된 CSV가 포함되어 있고, 파일이 없을 경우
            preprocessing 모듈이 UCI 원본 파일에서 결정론적으로 다시 생성할 수 있다.

            재현성은 구현의 일부로 다뤘다. train/test split은 stratify=y, test_size=0.2,
            random_state=42로 고정했다. imputation, IQR clipping, scaling, one-hot encoding,
            feature selection처럼 데이터로부터 학습되는 모든 변환은 train split 이후의 sklearn
            Pipeline 내부 또는 cross-validation fold 내부에서만 fit되도록 구성했다.
            """,
            FOOTER,
        )
        add_eda_page(pdf, report_dataset, FOOTER)
        add_text_page(
            pdf,
            "EDA 핵심 결과와 전처리 결정",
            """
            EDA notebook에서는 head, info, describe, target 분포, 결측률, 중복, 임상 범위
            점검, 그리고 age, trestbps, chol, thalach, oldpeak의 IQR 이상치를 확인했다. target은
            아주 심한 불균형은 아니지만 한쪽으로 기울어져 있어 단순 accuracy만 보면 임상적으로
            중요한 오류를 놓칠 수 있다. 그래서 모든 후보 모델에 대해 balanced accuracy,
            precision, recall, F1, confusion matrix를 함께 보고했다.

            통합 UCI 파일에는 여러 임상 변수의 결측이 남아 있다. 네 기관이 동일한 원 schema의
            모든 변수를 기록하지 않았기 때문이다. 이 결측 열이나 행을 단순 삭제하면 통합
            데이터의 장점을 잃게 되므로, 연속형 변수에는 median imputation을, 코드형 범주
            변수에는 most-frequent imputation을 적용했다.

            연속형 변수는 scaling 전에 train-fold에서 계산한 IQR 범위로 clipping한다. 이렇게
            하면 극단값이 StandardScaler의 평균과 표준편차를 과도하게 흔드는 것을 줄이면서도
            새 입력을 동일한 pipeline으로 처리할 수 있다. 코드형 범주 변수는
            handle_unknown='ignore'를 둔 one-hot encoding을 사용해, 학습 fold에 없던 category가
            추론 시 들어와도 파이프라인이 깨지지 않게 했다. 중복 행과 target이 빈 행은 split
            전에 제거했다.
            """,
            FOOTER,
        )
        add_table_page(
            pdf,
            "MLflow 실험 기반 모델 비교",
            model_comparison_for_report(comparison),
            "표 1. 각 run은 parameter, 5-fold CV metric, test metric, 선택 특성, confusion matrix, 학습된 model artifact를 MLflow에 기록한다. 최종 모델은 최고 test balanced accuracy와 0.03 이내인 후보 중 recall, F1, balanced accuracy 순으로 선택했다.",
            FOOTER,
            font_size=8.0,
        )
        add_final_model_page(
            pdf,
            final_summary,
            ARTIFACTS_DIR / f"{final_run}_confusion_matrix.png",
            final_confusion_counts,
            FOOTER,
        )
        add_text_page(
            pdf,
            "테스트, 패키징, CI, Repo 구성",
            """
            unittest는 작지만 실제로 깨질 수 있는 지점을 겨냥했다. 예측 결과의 row 수가 입력
            row 수와 일치하는지, predict_proba 값이 [0, 1] 범위에 있고 행별 합이 1인지,
            cholesterol이 허용 임상 범위를 벗어나면 거부되는지, 같은 seed의 두 pipeline이 같은
            입력에 대해 같은 예측을 내는지를 확인한다.

            Dockerfile은 python:3.10-slim을 기반으로 pinned requirements를 설치하고, source,
            data, artifacts, final model을 복사한 뒤 data/sample_input.csv에 대한 batch inference를
            실행한다. GitHub Actions workflow는 push와 pull request마다 의존성을 설치하고
            python -m unittest를 실행한다.

            Repo에는 과제 체크리스트의 핵심 파일을 모두 포함했다: data, notebooks/01_eda_preprocessing.ipynb,
            src/preprocessing.py, src/train.py, src/inference.py, src/monitor.py, tests/test_pipeline.py,
            mlruns, Dockerfile, requirements.txt, .github/workflows/ci.yml, report.pdf, README.md. README는
            설치, 학습, 추론, 테스트, 모니터링, 보고서 재생성, Docker build 순서를 한 번에 따라갈
            수 있게 정리했다.

            feature store 후보로는 최대 심박수 변수 thalach가 적절하다. 학습, 추론, drift 점검에
            반복적으로 쓰이므로 schema와 freshness 검증 이점이 크다. model registry에는
            selected_features를 남기고 싶다. 어떤 변환 특성이 SelectFromModel 이후 살아남았는지
            보여 주기 때문에, 모델 버전 비교와 감사 가능성을 높인다.
            """,
            FOOTER,
        )
        add_table_page(
            pdf,
            "KS 검정 기반 데이터 드리프트",
            drift,
            "표 2. monitor.py는 test set 복사본에서 cholesterol과 oldpeak를 인위적으로 이동시킨 뒤, 각 연속형 변수에 scipy.stats.ks_2samp를 적용하고 p < 0.05를 drift로 표시한다. 이동시키지 않은 변수들이 flag되지 않는 것도 sanity check 역할을 한다.",
            FOOTER,
        )
        add_table_page(
            pdf,
            "원본 테스트셋과 드리프트 테스트셋 성능 비교",
            performance,
            "표 3. 합성 input drift가 곧바로 모델의 임상적 무효를 의미하지는 않는다. 다만 balanced accuracy 하락은 계속 사용하기 전 재검토가 필요하다는 명확한 신호다.",
            FOOTER,
        )
        add_image_page(
            pdf,
            "시간에 따른 모니터링 지표",
            ARTIFACTS_DIR / "drift_performance_timeseries.png",
            "그림 3. 합성 timestamp에 따라 cholesterol shift를 키웠을 때 balanced accuracy가 어떻게 변하는지 보여 준다. 운영 환경에서는 같은 방식으로 model version과 data window별 성능을 추적한다.",
            FOOTER,
        )
        add_text_page(
            pdf,
            "서빙, 재학습, 윤리, AI 사용 공개",
            """
            이 프로젝트는 on-device보다 Model-as-a-Service 형태로 서빙하는 편이 적절하다고
            판단했다. MaaS는 모델 업데이트, audit logging, monitoring, rollback을 단순하게 만든다.
            이 과제 수준에서는 작은 latency 이점보다 버전 관리와 감시 가능성이 더 중요하다. PHI는
            전송 암호화, 엄격한 접근 제어, 최소 보관, 로그 redaction으로 보호해야 한다. On-device는
            데이터 이동을 줄일 수 있지만, 모니터링과 긴급 업데이트가 어려워진다.

            재학습은 확인된 drift와 사람이 검토한 label이 함께 있을 때, labeled follow-up data에서
            성능 저하가 지속될 때, 또는 정기 임상 review 주기에 맞춰 수행한다. Human-in-the-loop
            검토는 새 label이 training data에 들어가기 전과 새 model version을 올리기 전에 필요하다.
            모델이 영향을 준 진료 결정이 그대로 미래 label에 섞이면 runaway feedback loop가
            생길 수 있기 때문이다.

            가장 큰 한계는 데이터셋이다. UCI Heart Disease는 작고 오래되었으며 현대의 다양한 환자
            집단을 충분히 대표하지 않는다. 시간이 한 주 더 있었다면 calibration 분석, subgroup별
            성능 점검, 더 명확한 data card, 그리고 임상의가 이해할 수 있는 위험 요인 설명 화면을
            추가했을 것이다.

            AI 사용 공개. AI 도구는 boilerplate 구현, debugging, 문서 초안 정리에 사용했다. 최종
            코드, 실험 선택, 결과 해석, 제출물에 대한 책임은 본인에게 있다.
            """,
            FOOTER,
        )

    return REPORT_PATH


if __name__ == "__main__":
    path = generate_report()
    print(f"Generated {path}")
````

## tests\__init__.py

````python
"""Unit tests for CardioCare."""
````

## tests\test_pipeline.py

````python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    DATA_PATH,
    RANDOM_STATE,
    build_preprocessor,
    load_dataset,
    split_features_target,
    validate_feature_ranges,
)


def build_test_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "feature_selection",
                SelectFromModel(
                    RandomForestClassifier(
                        n_estimators=80,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                    threshold="median",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_dataset(DATA_PATH)
        X, y = split_features_target(df)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )

    def test_prediction_shape_matches_input_shape(self):
        pipeline = build_test_pipeline()
        pipeline.fit(self.X_train, self.y_train)
        predictions = pipeline.predict(self.X_test.head(12))
        self.assertEqual(predictions.shape[0], self.X_test.head(12).shape[0])

    def test_predict_proba_range_and_row_sum(self):
        pipeline = build_test_pipeline()
        pipeline.fit(self.X_train, self.y_train)
        probabilities = pipeline.predict_proba(self.X_test.head(12))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(probabilities.shape[0]))

    def test_clinical_range_validation_rejects_invalid_cholesterol(self):
        invalid = self.X_test.head(1).copy()
        invalid.loc[invalid.index[0], "chol"] = 900
        with self.assertRaises(ValueError):
            validate_feature_ranges(invalid)

    def test_fixed_seed_pipeline_is_deterministic(self):
        first = build_test_pipeline()
        second = build_test_pipeline()
        first.fit(self.X_train, self.y_train)
        second.fit(self.X_train, self.y_train)
        first_predictions = first.predict(self.X_test.head(20))
        second_predictions = second.predict(self.X_test.head(20))
        np.testing.assert_array_equal(first_predictions, second_predictions)


if __name__ == "__main__":
    unittest.main()
````

