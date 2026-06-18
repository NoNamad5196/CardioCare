# CardioCare 프로젝트 증거 요약

이 파일은 보고서 작성에 필요한 프로젝트 결과와 산출물 위치를 정리한 것입니다.

## 데이터

- 사용 데이터: UCI Heart Disease 통합 데이터셋
- 최종 CSV: `data/heart_disease.csv`
- 행 수: 918
- 입력 특성 수: 13
- 타깃: `target`
- 이진화 방식: `0 = 정상`, `1 = 심장병 있음`
- 샘플 추론 입력: `data/sample_input.csv`

## 전처리와 누수 방지

- train/test split: `test_size=0.2`, `stratify=y`, `random_state=42`
- 모든 학습되는 변환은 sklearn `Pipeline` 내부에 배치
- 연속형 특성: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- 연속형 처리: IQR clipping, median imputation, `StandardScaler`
- 범주형/코드형 처리: most-frequent imputation, one-hot encoding
- 특성 선택: `SelectFromModel(RandomForestClassifier(random_state=42))`
- feature selector와 scaler는 train fold 안에서만 fit되므로 test leakage를 피함

## 모델링

학습 및 비교한 모델 계열:

- Logistic Regression
- SVC
- Random Forest
- SVC hyperparameter tuning via `GridSearchCV`

MLflow 기록 내용:

- parameters
- 5-fold cross-validation metrics
- test metrics
- confusion matrix CSV/PNG
- selected features
- fitted model artifact
- model family tag

## 최종 모델

- selected run: `baseline_svc`
- model family: `svc`
- model version: `cardiocare-1.0`
- selected feature count: 14
- test balanced accuracy: 0.8387
- test precision: 0.8476
- test recall: 0.8725
- test F1: 0.8599

최종 선택 기준은 balanced accuracy가 최고 성능에서 0.03 이내인 후보 중 recall을 우선하고, 이후 F1과 balanced accuracy를 보는 방식입니다. 임상 보조 맥락에서는 false negative가 환자 follow-up 지연으로 이어질 수 있으므로 recall을 중요하게 보았습니다.

## 모델 비교 표 원본

`artifacts/model_comparison.csv`와 `report_materials/model_comparison.csv`를 사용하면 됩니다.

핵심 수치:

| run | balanced accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| baseline_logistic_regression | 0.8301 | 0.8515 | 0.8431 | 0.8473 |
| baseline_svc | 0.8387 | 0.8476 | 0.8725 | 0.8599 |
| baseline_random_forest | 0.8082 | 0.8091 | 0.8725 | 0.8396 |
| tuned_svc | 0.8387 | 0.8476 | 0.8725 | 0.8599 |

## 추론

실행 명령:

```bash
python src/inference.py --input data/sample_input.csv --output artifacts/predictions.csv
```

추론 output:

- `artifacts/predictions.csv`
- `report_materials/predictions.csv`

추론 로그에는 timestamp, model version, input shape, predictions, 가능한 경우 actual labels가 포함됩니다.

## 테스트

테스트 파일: `tests/test_pipeline.py`

테스트 항목:

- prediction shape
- probability range와 row-wise sum
- `chol` 입력 범위 검증
- fixed seed deterministic prediction

검증 결과:

```text
python -m unittest
Ran 4 tests
OK
```

## Docker

Dockerfile은 `python:3.10-slim` 기반입니다.

검증한 명령:

```bash
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

컨테이너는 `data/sample_input.csv`를 입력으로 사용해 `artifacts/docker_predictions.csv`에 예측 결과를 저장합니다.

## 모니터링과 드리프트

실행 명령:

```bash
python src/monitor.py
```

드리프트 시뮬레이션:

- test set 복사본에서 `chol`과 `oldpeak` 분포를 이동
- 각 연속형 특성에 대해 train distribution vs shifted test distribution으로 `ks_2samp` 실행
- `p < 0.05`이면 drift flag

드리프트 결과:

| feature | p-value | drift flag |
| --- | ---: | --- |
| age | 0.8793 | False |
| trestbps | 0.9999 | False |
| chol | 3.7648e-19 | True |
| thalach | 0.7058 | False |
| oldpeak | 4.7791e-127 | True |

성능 비교:

- original test balanced accuracy: 0.8387
- drifted test balanced accuracy: 0.6280

## 보고서 그림/표 자료

`report_materials/`에서 사용하면 됩니다.

- `01_eda_summary.png`: EDA 요약 그림
- `02_model_comparison_table.png`: 모델 비교 표 이미지
- `03_drift_report_table.png`: KS drift 결과 표 이미지
- `04_drift_performance_table.png`: 원본 vs drifted 성능 비교 표
- `baseline_svc_confusion_matrix.png`: 최종 모델 confusion matrix
- `drift_performance_timeseries.png`: drift 전후 성능 변화 시계열

## 서빙과 재학습 전략

보고서에서는 Model-as-a-Service를 선택하는 쪽이 현재 구현과 잘 맞습니다.

- 지연 시간: 작은 tabular model이라 서버 추론 지연은 관리 가능
- PHI/개인정보: 서버에서 접근 제어, 암호화, audit logging을 중앙 관리하기 쉬움
- 업데이트 주기: 모델 교체, rollback, 모니터링을 중앙에서 운영하기 쉬움

재학습은 drift flag만으로 자동 수행하지 않고, drift 지속성, 성능 저하, 사람 검토를 거친 새 label 확보를 함께 확인한 뒤 진행합니다. 모델 promotion 전에는 Human-in-the-loop review가 필요합니다.

## 윤리와 한계

- 교육용 prototype이며 실제 의료기기가 아님
- 작은 공개 데이터셋 기반이라 외부 검증 필요
- 실제 적용 전 calibration, fairness, subgroup performance, PHI 보호 설계 필요
- CardioCare는 위험 신호를 "알릴" 수 있지만 진단을 "결정"해서는 안 됨

