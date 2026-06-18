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

