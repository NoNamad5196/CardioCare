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
