# CardioCare

CardioCare는 UCI Heart Disease 데이터를 이용해 심장병 가능성을 이진 분류하는 종단간 머신러닝 프로젝트입니다. 이 프로젝트의 목적은 심장 전문의의 판단을 보조하는 것이며, 모델이 진단이나 치료 결정을 단독으로 내리는 시스템이 아닙니다.

GitHub 저장소: https://github.com/NoNamad5196/CardioCare

## 1. 프로젝트 개요

- 문제: 환자 임상 지표를 바탕으로 심장병 여부를 예측합니다.
- 데이터: UCI Heart Disease 데이터셋의 통합 버전, 총 918행과 13개 입력 특성 사용.
- 타깃: 원래 다중 클래스인 `num` 값을 `0 = 정상`, `1 = 심장병 있음`으로 이진화했습니다.
- 최종 모델: `SVC`
- 모델 버전: `cardiocare-1.0`
- 주요 산출물: 학습 코드, 추론 코드, 모니터링 코드, unittest, Dockerfile, GitHub Actions CI, MLflow 실행 기록, 최종 보고서.

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
tests/
  test_pipeline.py
artifacts/
models/
mlruns/
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

## 10. Feature Store와 Model Registry 메모

Feature Store 후보로는 `thalach`가 적절합니다. 최대 심박수는 학습, 추론, 드리프트 모니터링에서 반복적으로 사용되므로 schema, freshness, validation을 명시적으로 관리할 가치가 있습니다.

Model Registry에 기록해야 할 주요 메타데이터는 `selected_features`입니다. 어떤 변환 특성이 최종적으로 살아남았는지 남겨야 모델 버전 간 비교와 감사 가능성이 좋아집니다.

## 11. 서빙과 재학습 전략

이 프로젝트에서는 on-device serving보다 Model-as-a-Service가 더 적합하다고 판단했습니다. 서버 기반 서빙은 모델 업데이트, rollback, 감사 로그, 모니터링을 중앙에서 관리하기 쉽습니다. 단, 실제 의료 환경에서는 PHI 보호를 위해 암호화된 전송, 엄격한 접근 제어, 최소 보존 정책이 필요합니다.

재학습은 단순히 drift가 한 번 감지되었다는 이유만으로 자동 수행하지 않습니다. drift 감지, 성능 저하, 사람의 검토를 거친 새로운 label 확보가 함께 확인될 때 재학습 후보로 올리고, 모델 promotion 전에는 Human-in-the-loop 검토가 필요합니다.

## 12. 한계와 윤리

CardioCare는 오래된 공개 데이터셋을 기반으로 한 교육용 prototype입니다. 실제 임상 적용에는 외부 검증, calibration, 공정성 분석, 개인정보 보호 설계, 의료진 검토 절차가 추가로 필요합니다. 모델은 위험 신호를 알릴 수는 있지만, 진단과 치료 결정을 대신해서는 안 됩니다.
