# CardioCare

CardioCare is an end-to-end machine learning project for binary heart disease prediction using the UCI Heart Disease dataset. It is a clinical decision-support prototype only: it can inform a cardiologist's review, but it must not make diagnosis or treatment decisions by itself.

## Reproduce

Use Python 3.10 or newer. A virtual environment is recommended so the `python` and `pip` commands below point to the same interpreter.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
python src/train.py
python src/inference.py --input data/sample_input.csv
python -m unittest
python src/monitor.py
python src/report.py
docker build -t cardiocare:1.0 .
```

Docker Desktop or another Docker daemon must be running before the `docker build` command.

The repository includes `data/heart_disease.csv`. If it is missing, `src/preprocessing.py` downloads the four UCI processed files from:

- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

The multi-class UCI target is binarized as `target = 0` for no disease and `target = 1` for any disease label greater than zero.

## Project Structure

```text
data/
notebooks/01_eda_preprocessing.ipynb
src/preprocessing.py
src/train.py
src/inference.py
src/monitor.py
src/report.py
tests/test_pipeline.py
requirements.txt
Dockerfile
.github/workflows/ci.yml
README.md
report.pdf
```

Generated outputs:

- `models/final_model.pkl`
- `models/final_model_metadata.json`
- `mlruns/`
- `artifacts/model_comparison.csv`
- `artifacts/*_confusion_matrix.csv`
- `artifacts/drift_report.csv`
- `artifacts/drift_performance_timeseries.png`
- `logs/inference.log`
- `logs/monitor.log`

## Data Leakage Controls

All learned transformations are inside sklearn `Pipeline` objects. The code splits data first with `train_test_split(test_size=0.2, stratify=y, random_state=42)`, then fits preprocessing inside cross-validation folds and the final train split only.

The model pipeline order is:

```text
ColumnTransformer preprocessing
-> SelectFromModel(RandomForestClassifier(random_state=42))
-> classifier
```

Continuous features use train-fold IQR clipping, median imputation, and `StandardScaler`. Categorical-coded features use most-frequent imputation and one-hot encoding. `StandardScaler` is never fit on test data.

## Training and MLflow

`python src/train.py` trains and compares:

- Logistic Regression
- SVC
- Random Forest

Each model run logs parameters, 5-fold CV metrics, test metrics, selected features, confusion matrix artifacts, and the fitted model artifact to MLflow. At least one strongest baseline candidate is tuned with `GridSearchCV`.

Reported metrics:

- balanced accuracy
- precision
- recall
- F1
- confusion matrix

Final model selection prioritizes recall among candidates within 0.03 balanced accuracy of the best test result, then F1, then balanced accuracy. This reflects the clinical cost of false negatives.

## Inference

```bash
python src/inference.py --input data/sample_input.csv --output artifacts/predictions.csv
```

The input CSV must contain the 13 feature columns. If a `target` column is present, it is copied into the output and included in the inference log.

Logged fields include timestamp, model version, input shape, predictions, and actual labels when available.

## Monitoring

```bash
python src/monitor.py
```

Monitoring recreates the deterministic train/test split, shifts cholesterol and oldpeak in a test-set copy, and runs `scipy.stats.ks_2samp` for each continuous feature. Features with `p < 0.05` are flagged for drift.

The script saves:

- `artifacts/drift_report.csv`
- `artifacts/drift_performance_comparison.csv`
- `artifacts/performance_timeseries.csv`
- `artifacts/drift_performance_timeseries.png`
- `logs/monitor.log`

## Tests and CI

```bash
python -m unittest
```

The unittest suite checks:

- prediction shape matches input rows
- predicted probabilities are in `[0, 1]` and sum to 1 per row
- invalid cholesterol range is rejected
- fixed-seed pipeline predictions are deterministic

GitHub Actions runs the unit tests on every push and pull request using Python 3.10.

## Docker

Build:

```bash
docker build -t cardiocare:1.0 .
```

Run:

```bash
docker run --rm cardiocare:1.0
```

The image runs batch inference on `data/sample_input.csv` using `models/final_model.pkl`.

## Feature Store and Registry Notes

A useful feature-store candidate is `thalach` because maximum heart rate is repeatedly used during training, inference, and drift monitoring. Keeping it in a feature store would make schema, freshness, and validation checks explicit.

A model-registry metadata field that should be recorded is `selected_features`. It explains what transformed features survived `SelectFromModel`, supports auditability, and helps reviewers compare model versions beyond headline metrics.

## Serving and Retraining Strategy

For this project, Model-as-a-Service is preferred over on-device serving. It simplifies audit logging, version rollback, and monitoring. PHI protection would require encrypted transport, strict access control, and minimal retention.

Retraining should happen after confirmed drift plus human-reviewed labels, sustained performance degradation, or a scheduled review. Human-in-the-loop review is required before model promotion to avoid feedback loops where model-influenced decisions contaminate future labels.

## Ethics

CardioCare is an educational prototype built from a small, dated public dataset. It can surface risk signals, but it cannot replace clinical judgment. False negatives are especially important because they may delay needed follow-up care.
