# CardioCare 보고서 작성용 자료

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
