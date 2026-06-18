# CardioCare 과제 요구사항 요약

이 파일은 `ML_기말프로젝트_CardioCare.pdf`의 보고서 작성 및 제출 요구사항을 보고서 작성자가 빠르게 확인할 수 있도록 요약한 것입니다. 최종 판단 기준은 반드시 원본 PDF입니다.

## 프로젝트 목표

CardioCare는 임상 데이터로 심장병 가능성을 예측해 심장 전문의의 의사결정을 보조하는 종단간 머신러닝 시스템입니다. 핵심 윤리 관점은 "알리되, 결정하지 않는다(inform, not decide)"입니다.

## 필수 구현 범위

1. 문제 정의, EDA, 데이터 전처리
2. 특성 공학, 모델 학습, 실험 관리, 모델 평가
3. unittest 기반 테스트, Docker 패키징, CI 워크플로
4. logging 기반 추론 계측, KS 검정 기반 드리프트 탐지, 서빙 및 재학습 전략 문서화
5. GitHub repository와 6-10쪽 PDF 보고서 제출

## 데이터 요구사항

- UCI Heart Disease 데이터셋 사용
- Cleveland subset만 사용해도 되지만 통합 버전 권장
- 어떤 데이터 버전을 사용했는지 명시
- 다중 클래스 target은 `0 = 정상`, `1 = 심장병`으로 이진화
- 데이터를 결정론적으로 불러오거나 다운로드할 수 있어야 함

## 도구 요구사항

- Python 3.10+
- pandas, numpy, scikit-learn
- matplotlib 또는 seaborn
- MLflow
- unittest
- `scipy.stats.ks_2samp`
- logging
- Dockerfile과 Docker image build
- GitHub Actions 등 CI 도구 1종

## 보고서 필수 섹션

보고서는 PDF 단일 파일, 6-10쪽이어야 하며 다음 내용을 포함해야 합니다.

1. 문제 정의와 사용 목적: CardioCare는 의사결정을 보조할 뿐 단독 결정하지 않음을 명시
2. EDA 핵심 결과: 그림은 2-3개 정도로 제한
3. EDA에 근거한 전처리 결정
4. MLflow 실험 기반 모델 비교 표와 최종 선택 정당화
5. 테스트와 패키징: 무엇을 왜 테스트했는지 설명
6. 드리프트 결과와 재학습/피드백 루프 계획: 폭주하는 feedback loop 위험과 Human-in-the-loop 지점 명시
7. 서빙 선택: Model-as-a-Service 또는 On-Device 중 하나를 지연 시간, PHI/개인정보, 업데이트 주기 관점에서 정당화
8. 한계, 윤리적 고려사항, 시간이 1주 더 있다면 할 일

## 채점자가 확인할 재현성 기준

채점자는 repository를 clone한 뒤 README를 따라 다음을 확인할 수 있어야 합니다.

- 의존성 설치
- `python src/train.py`
- MLflow에서 3개 이상 실행과 지표/artifact 확인
- `python -m unittest`
- `docker build`와 `docker run`
- `python src/monitor.py` 실행 시 drift flag와 성능 저하 확인

## 자동 감점 주의

- README 부재 또는 재현 불가: -10
- 데이터 누수 발견: -10
- AI 도구 사용 미공개 또는 출처 없는 코드 복사: -10
- 저장소 구조 미준수: -5

