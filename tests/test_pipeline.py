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
