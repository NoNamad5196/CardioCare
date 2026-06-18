# CardioCare Claude 보고서 패키지

이 패키지는 Claude에게 최종 보고서 초안을 맡길 때 원본 과제 PDF와 함께 업로드하기 위한 자료입니다.

## 업로드할 파일

1. `ML_기말프로젝트_CardioCare.pdf`
2. `CardioCare_claude_report_package.zip`

## ZIP 안의 구성

- `CLAUDE_REPORT_PROMPT.md`: Claude에 그대로 붙여넣을 프롬프트
- `ASSIGNMENT_REQUIREMENTS_SUMMARY.md`: 과제 요구사항 요약
- `PROJECT_EVIDENCE_SUMMARY.md`: 프로젝트 결과와 보고서 근거 요약
- `FULL_CODE_CONTEXT.md`: 주요 코드 전문을 Markdown으로 합친 파일
- `code/`: 실제 코드와 설정 파일
- `evidence/`: 보고서에 사용할 표, 그림, CSV, JSON
- `data_samples/`: 데이터 CSV와 샘플 입력

## 사용 방법

Claude에 원본 과제 PDF와 이 ZIP 파일을 업로드한 뒤, `CLAUDE_REPORT_PROMPT.md` 내용을 그대로 붙여넣으면 됩니다. 보고서의 수치와 설명은 반드시 ZIP 안의 evidence 및 code 자료를 근거로 작성하도록 요청해 두었습니다.

