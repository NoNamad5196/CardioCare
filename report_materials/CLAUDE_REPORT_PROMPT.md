# Claude 보고서 작성 프롬프트

아래 프롬프트를 Claude에 그대로 붙여넣고, 함께 업로드할 파일은 다음 두 가지입니다.

1. 원본 과제 PDF: `ML_기말프로젝트_CardioCare.pdf`
2. 프로젝트 자료 ZIP: `CardioCare_claude_report_package.zip`

---

너는 컴퓨터공학과 4학년 머신러닝 수업의 기말 프로젝트 보고서를 다듬어 주는 조교 역할이다. 내가 업로드한 원본 과제 PDF와 `CardioCare_claude_report_package.zip` 안의 프로젝트 자료를 근거로, CardioCare 프로젝트의 최종 보고서 초안을 한국어로 작성해 줘.

중요한 원칙:

- 보고서는 6-10쪽 PDF로 옮기기 좋은 분량과 구조로 작성한다.
- 문체는 자연스러운 대학생 보고서 스타일로 쓴다. 지나치게 광고문처럼 쓰거나 AI가 쓴 듯한 포괄적 문장을 반복하지 말고, 코드와 실험 결과에 근거한 구체적인 문장으로 쓴다.
- 과제의 윤리 요구사항을 반드시 반영한다. CardioCare는 "알리되, 결정하지 않는" 임상 의사결정 보조 도구이며, 진단이나 치료 결정을 단독으로 수행하지 않는다고 명확히 쓴다.
- 결과 수치는 자료 ZIP 안의 CSV/JSON/PNG와 코드에 있는 값만 사용한다. 자료에 없는 성능이나 실험은 절대 만들어내지 않는다.
- 모든 핵심 주장에는 어떤 코드나 산출물에서 확인되는지 자연스럽게 연결해 설명한다. 예: `src/train.py`, `artifacts/model_comparison.csv`, `src/monitor.py`, `tests/test_pipeline.py`.
- AI 사용 공개는 과제 감점 항목과 관련되므로 숨기지 않는다. 보고서 마지막 부록 또는 한계 섹션에 1문단으로, AI 도구는 초안 정리와 문장 개선, 디버깅 보조에 사용했으며 최종 코드와 결과 해석의 책임은 작성자에게 있다는 식으로 자연스럽게 작성한다.

자료 ZIP에서 우선 읽을 파일:

- `ASSIGNMENT_REQUIREMENTS_SUMMARY.md`: 과제 요구사항 요약
- `PROJECT_EVIDENCE_SUMMARY.md`: 프로젝트 결과와 실험 증거 요약
- `FULL_CODE_CONTEXT.md`: 주요 코드 전문
- `code/`: 실제 코드 파일
- `evidence/`: 보고서에 넣을 PNG, CSV, JSON 자료
- `data_samples/`: 데이터와 샘플 입력

보고서 구조는 다음 순서를 따른다.

1. 문제 정의와 사용 목적
   - 심장병 가능성 예측 문제를 정의한다.
   - CardioCare가 심장 전문의의 판단을 보조하는 도구임을 밝힌다.
   - false negative가 왜 중요한지 간단히 설명한다.

2. 데이터와 EDA 핵심 결과
   - UCI Heart Disease 통합 데이터셋, 918행, 13개 입력 특성, 이진 target을 설명한다.
   - target 분포, 결측/이상치, 연속형 특성의 분포를 중심으로 EDA 결과를 요약한다.
   - 그림은 `evidence/01_eda_summary.png`를 중심으로 1-2개만 추천한다.

3. 전처리 결정
   - `src/preprocessing.py`와 `src/train.py`를 근거로 train/test split 이후 Pipeline 안에서 imputer, scaler, feature selector가 fit된다는 점을 설명한다.
   - 연속형 특성에는 IQR clipping, median imputation, StandardScaler를 사용하고, 범주형/코드형 특성에는 most-frequent imputation과 one-hot encoding을 사용했다고 쓴다.
   - 데이터 누수를 피한 설계를 명확히 설명한다.

4. 특성 공학과 모델 실험
   - Logistic Regression, SVC, Random Forest, tuned SVC를 비교했다고 설명한다.
   - MLflow에 파라미터, CV 지표, test 지표, confusion matrix, selected features, model artifact를 기록했다고 쓴다.
   - `evidence/02_model_comparison_table.png` 또는 `evidence/model_comparison.csv`를 표로 활용한다.

5. 최종 모델 선택
   - 최종 모델은 `baseline_svc`이며, model family는 `svc`라고 쓴다.
   - 핵심 성능: balanced accuracy 0.8387, precision 0.8476, recall 0.8725, F1 0.8599.
   - 선택 기준은 balanced accuracy가 최고 성능에서 0.03 이내인 후보 중 recall을 우선하고, 이후 F1과 balanced accuracy를 본 것이라고 설명한다.
   - 임상 보조 맥락에서 false negative를 줄이는 방향이 중요하다고 연결한다.

6. 테스트, 패키징, CI
   - `tests/test_pipeline.py`의 4개 unittest를 설명한다.
   - prediction shape, probability range/sum, `chol` 입력 범위, deterministic prediction을 각각 왜 테스트했는지 쓴다.
   - Dockerfile은 `python:3.10-slim` 기반이며 샘플 입력 추론을 실행한다고 쓴다.
   - `.github/workflows/ci.yml`에서 push와 pull request마다 `python -m unittest`가 실행된다고 설명한다.

7. 모니터링과 드리프트
   - `src/inference.py`의 logging 항목과 `src/monitor.py`의 KS 검정 방식을 설명한다.
   - test set 복사본에서 `chol`과 `oldpeak`를 인위적으로 이동시켰고, KS 검정에서 두 특성이 drift로 flag되었다고 쓴다.
   - original balanced accuracy 0.8387, drifted balanced accuracy 0.6280을 비교해 입력 drift와 성능 저하의 관계를 설명한다.
   - `evidence/03_drift_report_table.png`, `evidence/04_drift_performance_table.png`, `evidence/drift_performance_timeseries.png` 중 필요한 것만 사용하도록 제안한다.

8. 서빙과 재학습 전략
   - 현재 프로젝트에는 Model-as-a-Service가 적합하다고 정리한다.
   - 지연 시간, PHI/개인정보, 업데이트 주기 관점에서 정당화한다.
   - drift flag만으로 자동 재학습하지 않고, 지속적 drift, 성능 저하, 사람 검토를 거친 label 확보 뒤 재학습 후보로 올린다고 쓴다.
   - 모델 promotion 전 Human-in-the-loop review가 필요하며, 모델이 만든 예측이 다시 label에 영향을 주는 feedback loop 위험을 설명한다.

9. 한계, 윤리, 추가 개선
   - 오래된 공개 데이터셋 기반의 교육용 prototype이라는 한계를 쓴다.
   - 실제 의료 적용 전 외부 검증, calibration, subgroup/fairness 분석, PHI 보호 설계가 필요하다고 쓴다.
   - 1주가 더 있다면 calibration curve, subgroup recall, MLflow model registry, 간단한 API serving, 더 현실적인 drift simulation을 추가하겠다고 제안한다.

10. AI 사용 공개
   - 부록 또는 마지막 단락에 짧게 작성한다.
   - 예시 톤: "본 프로젝트에서는 AI 도구를 코드 디버깅, 보고서 초안 구조화, 문장 다듬기 보조에 사용하였다. 실험 실행, 산출물 확인, 최종 모델 선택과 해석의 책임은 작성자에게 있으며, 보고서의 수치와 결론은 repository의 코드 및 산출물에 근거한다."

출력 형식:

- 한국어 보고서 초안으로 작성한다.
- 제목과 section heading을 포함한다.
- 표나 그림이 들어갈 위치는 `[그림 1: ... 삽입]`, `[표 1: ... 삽입]`처럼 표시한다.
- 너무 완벽한 홍보 문구보다, 과제 수행 과정을 설명하는 담백한 보고서 문체를 유지한다.
- 마지막에 "보고서에 넣을 추천 그림/표 목록"을 5개 이하로 따로 정리한다.

