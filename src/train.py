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
