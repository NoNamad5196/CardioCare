from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cardiocare")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORT_PATH = PROJECT_ROOT / "report.pdf"
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
FOOTER = "CardioCare 기계학습 기말 프로젝트 | UCI Heart Disease | 알리되, 결정하지 않는다"
KOREAN_FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")


def configure_fonts() -> None:
    if KOREAN_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(KOREAN_FONT_PATH))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42


def wrap_lines(text: str, width: int = 92) -> str:
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        collapsed = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if collapsed:
            paragraphs.append(collapsed)
    return "\n\n".join(
        textwrap.fill(paragraph, width=width, break_on_hyphens=False)
        for paragraph in paragraphs
    )


def format_metric(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def add_text_page(pdf: PdfPages, title: str, body: str, footer: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=18, fontweight="bold", color="#18314f")
    fig.text(
        0.08,
        0.89,
        wrap_lines(body, width=62),
        fontsize=10.5,
        va="top",
        linespacing=1.45,
        color="#1f2933",
    )
    fig.text(0.08, 0.045, footer, fontsize=8.5, color="#667085")
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
    note: str,
    footer: str,
    font_size: float = 8.5,
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color="#18314f", pad=18)
    display_df = dataframe.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.3f}")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e6eef8")
            cell.set_text_props(weight="bold", color="#18314f")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
    fig.text(0.06, 0.08, wrap_lines(note, width=108), fontsize=9.5, color="#344054")
    fig.text(0.06, 0.035, footer, fontsize=8.5, color="#667085")
    fig.tight_layout(rect=[0.04, 0.12, 0.96, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def add_eda_page(pdf: PdfPages, dataframe: pd.DataFrame, footer: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("전처리 결정을 위한 EDA 요약", x=0.06, y=0.94, ha="left", fontsize=17, fontweight="bold", color="#18314f")

    target_counts = dataframe["target"].value_counts(normalize=True).sort_index()
    axes[0].bar(["0: 정상", "1: 심장병"], target_counts.values, color=["#4c78a8", "#f58518"])
    axes[0].set_title("타깃 클래스 비율")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("비율")
    for idx, value in enumerate(target_counts.values):
        axes[0].text(idx, value + 0.025, f"{value:.1%}", ha="center", fontsize=9)

    missing_rates = dataframe.isna().mean().sort_values(ascending=False)
    missing_rates = missing_rates[missing_rates > 0].head(6)
    axes[1].barh(missing_rates.index[::-1], missing_rates.values[::-1], color="#72b7b2")
    axes[1].set_title("결측률 상위 변수")
    axes[1].set_xlabel("결측률")
    axes[1].set_xlim(0, max(0.55, float(missing_rates.max()) * 1.2 if len(missing_rates) else 0.55))
    for y_pos, value in enumerate(missing_rates.values[::-1]):
        axes[1].text(value + 0.01, y_pos, f"{value:.1%}", va="center", fontsize=9)

    outlier_rows = []
    for column in CONTINUOUS_FEATURES:
        series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_rows.append(((series < lower) | (series > upper)).sum())
    axes[2].bar(CONTINUOUS_FEATURES, outlier_rows, color="#54a24b")
    axes[2].set_title("IQR 이상치 개수")
    axes[2].tick_params(axis="x", rotation=35)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    note = (
        "그림 1. 통합 데이터는 약한 클래스 불균형, 수집 기관 차이에 따른 결측, "
        "연속형 변수의 이상치를 함께 보인다. 이 결과를 근거로 stratified split, "
        "balanced accuracy/recall 중심 평가, 대치, train-fold IQR clipping, scaling을 적용했다."
    )
    fig.text(0.06, 0.10, wrap_lines(note, width=105), fontsize=9.5, color="#344054")
    fig.text(0.06, 0.035, footer, fontsize=8.5, color="#667085")
    fig.tight_layout(rect=[0.04, 0.16, 0.96, 0.88])
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, note: str, footer: str) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.text(0.08, 0.94, title, fontsize=18, fontweight="bold", color="#18314f")
    if image_path.exists():
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_position([0.08, 0.25, 0.84, 0.56])
    else:
        fig.text(0.08, 0.55, f"Missing image: {image_path}", fontsize=11, color="#b42318")
    fig.text(0.08, 0.17, wrap_lines(note, width=62), fontsize=10, color="#344054")
    fig.text(0.08, 0.045, footer, fontsize=8.5, color="#667085")
    pdf.savefig(fig)
    plt.close(fig)


def add_final_model_page(
    pdf: PdfPages,
    final_summary: dict[str, str],
    image_path: Path,
    confusion_counts: dict[str, int] | None,
    footer: str,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, "최종 모델 선택 근거", fontsize=18, fontweight="bold", color="#18314f")

    metrics = "\n".join(
        [
            f"선택된 run: {final_summary['selected_run']}",
            f"모델 계열: {final_summary['model_family']}",
            f"테스트 balanced accuracy: {final_summary['test_balanced_accuracy']}",
            f"테스트 recall: {final_summary['test_recall']}",
            f"테스트 precision: {final_summary['test_precision']}",
            f"테스트 F1: {final_summary['test_f1']}",
        ]
    )
    fig.text(0.08, 0.885, metrics, fontsize=10.3, va="top", linespacing=1.42, color="#1f2933")

    if image_path.exists():
        ax = fig.add_axes([0.49, 0.57, 0.40, 0.27])
        ax.axis("off")
        ax.imshow(plt.imread(image_path))
    else:
        fig.text(0.50, 0.72, f"Missing image: {image_path.name}", fontsize=10, color="#b42318")

    count_sentence = "이 문제에서는 단일 accuracy보다 false negative 개수를 직접 확인하는 것이 더 중요하므로 confusion matrix를 함께 제시했다."
    if confusion_counts:
        count_sentence = (
            f"Confusion matrix 기준으로 실제 심장병 test record 중 {confusion_counts['fn']}건은 놓쳤고, "
            f"{confusion_counts['tp']}건은 올바르게 위험군으로 표시했다."
        )
    body = f"""
    최종 모델은 먼저 최고 test balanced accuracy와 0.03 이내인 후보를 남기고, 그 안에서 recall,
    F1, balanced accuracy 순으로 고르는 규칙을 따랐다. 심장병 가능성을 놓치는 false negative가
    추후 진료 지연으로 이어질 수 있기 때문에 recall을 우선순위에 둔 것이다.

    {count_sentence} 동시에 precision도 보고해 false positive로 인한 추가 검토 부담을 숨기지
    않았다. 이 모델은 교육용 의사결정 보조 도구이며, 점수는 차트를 다시 볼 이유이지 의사의
    판단을 대체할 이유가 아니다.

    그림 2. 최종 모델의 hold-out test confusion matrix.
    """
    fig.text(0.08, 0.50, wrap_lines(body, width=62), fontsize=10.2, va="top", linespacing=1.42, color="#1f2933")
    fig.text(0.08, 0.045, footer, fontsize=8.5, color="#667085")
    pdf.savefig(fig)
    plt.close(fig)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({"status": [f"Missing artifact: {path.name}. Run train.py and monitor.py."]})


def load_report_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, na_values=["?", ""])
    return pd.DataFrame(
        {
            "target": [0, 1],
            "age": [None, None],
            "trestbps": [None, None],
            "chol": [None, None],
            "thalach": [None, None],
            "oldpeak": [None, None],
        }
    )


def model_comparison_for_report(comparison: pd.DataFrame) -> pd.DataFrame:
    if "status" in comparison.columns:
        return comparison
    columns = {
        "run_name": "run",
        "model_family": "계열",
        "selected_feature_count": "특성수",
        "cv_balanced_accuracy_mean": "CV bAcc",
        "cv_precision_mean": "CV prec",
        "cv_recall_mean": "CV recall",
        "cv_f1_mean": "CV F1",
        "test_balanced_accuracy": "test bAcc",
        "test_precision": "test prec",
        "test_recall": "test recall",
        "test_f1": "test F1",
    }
    display = comparison[list(columns)].rename(columns=columns)
    display["run"] = display["run"].replace(
        {
            "baseline_logistic_regression": "base_logreg",
            "baseline_random_forest": "base_rf",
            "baseline_svc": "base_svc",
            "tuned_svc": "tuned_svc",
        }
    )
    display["계열"] = display["계열"].replace(
        {
            "logistic_regression": "logreg",
            "random_forest": "rf",
        }
    )
    return display


def final_model_summary(metadata: dict, comparison: pd.DataFrame) -> dict[str, str]:
    selected_run = metadata.get("selected_run", "baseline_random_forest")
    row = comparison.loc[comparison.get("run_name", pd.Series(dtype=str)) == selected_run]
    if row.empty:
        return {
            "selected_run": str(selected_run),
            "model_family": str(metadata.get("model_family", "unknown")),
            "test_balanced_accuracy": format_metric(metadata.get("metrics", {}).get("test_balanced_accuracy")),
            "test_recall": format_metric(metadata.get("metrics", {}).get("test_recall")),
            "test_precision": format_metric(metadata.get("metrics", {}).get("test_precision")),
            "test_f1": format_metric(metadata.get("metrics", {}).get("test_f1")),
        }

    row = row.iloc[0]
    return {
        "selected_run": str(row["run_name"]),
        "model_family": str(row["model_family"]),
        "test_balanced_accuracy": format_metric(row["test_balanced_accuracy"]),
        "test_recall": format_metric(row["test_recall"]),
        "test_precision": format_metric(row["test_precision"]),
        "test_f1": format_metric(row["test_f1"]),
    }


def confusion_counts_for_run(run_name: str) -> dict[str, int] | None:
    path = ARTIFACTS_DIR / f"{run_name}_confusion_matrix.csv"
    if not path.exists():
        return None
    matrix = pd.read_csv(path, index_col=0)
    return {
        "tn": int(matrix.loc["actual_0", "predicted_0"]),
        "fp": int(matrix.loc["actual_0", "predicted_1"]),
        "fn": int(matrix.loc["actual_1", "predicted_0"]),
        "tp": int(matrix.loc["actual_1", "predicted_1"]),
    }


def generate_report() -> Path:
    configure_fonts()
    metadata_path = PROJECT_ROOT / "models" / "final_model_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    comparison = read_optional_csv(ARTIFACTS_DIR / "model_comparison.csv")
    drift = read_optional_csv(ARTIFACTS_DIR / "drift_report.csv")
    performance = read_optional_csv(ARTIFACTS_DIR / "drift_performance_comparison.csv")
    report_dataset = load_report_dataset()
    final_summary = final_model_summary(metadata, comparison)
    final_run = final_summary["selected_run"]
    final_confusion_counts = confusion_counts_for_run(final_run)

    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(
            pdf,
            "CardioCare: 심장병 예측 종단간 ML 시스템",
            """
            CardioCare는 심장병 위험 신호를 추정하는 교육용 종단간 머신러닝 시스템이다. 이
            프로젝트의 기본 원칙은 "알리되, 결정하지 않는다"이다. 모델은 환자 record가
            심장병 위험과 얼마나 비슷한지 알려 주지만, 진단이나 치료 결정을 자동으로 내리는
            시스템이 아니다. 결과는 심장 전문의가 차트를 검토할 때 참고하는 보조 정보로만
            사용되어야 한다.

            데이터는 UCI Heart Disease의 네 개 processed 파일(Cleveland, Hungarian,
            Switzerland, VA)을 통합해 사용했다. 원래 target은 여러 질병 정도를 나타내는
            다중 클래스이지만, 이 과제에서는 target=0을 정상, target=1을 심장병 관측 기록으로
            이진화했다. repository에는 정리된 CSV가 포함되어 있고, 파일이 없을 경우
            preprocessing 모듈이 UCI 원본 파일에서 결정론적으로 다시 생성할 수 있다.

            재현성은 구현의 일부로 다뤘다. train/test split은 stratify=y, test_size=0.2,
            random_state=42로 고정했다. imputation, IQR clipping, scaling, one-hot encoding,
            feature selection처럼 데이터로부터 학습되는 모든 변환은 train split 이후의 sklearn
            Pipeline 내부 또는 cross-validation fold 내부에서만 fit되도록 구성했다.
            """,
            FOOTER,
        )
        add_eda_page(pdf, report_dataset, FOOTER)
        add_text_page(
            pdf,
            "EDA 핵심 결과와 전처리 결정",
            """
            EDA notebook에서는 head, info, describe, target 분포, 결측률, 중복, 임상 범위
            점검, 그리고 age, trestbps, chol, thalach, oldpeak의 IQR 이상치를 확인했다. target은
            아주 심한 불균형은 아니지만 한쪽으로 기울어져 있어 단순 accuracy만 보면 임상적으로
            중요한 오류를 놓칠 수 있다. 그래서 모든 후보 모델에 대해 balanced accuracy,
            precision, recall, F1, confusion matrix를 함께 보고했다.

            통합 UCI 파일에는 여러 임상 변수의 결측이 남아 있다. 네 기관이 동일한 원 schema의
            모든 변수를 기록하지 않았기 때문이다. 이 결측 열이나 행을 단순 삭제하면 통합
            데이터의 장점을 잃게 되므로, 연속형 변수에는 median imputation을, 코드형 범주
            변수에는 most-frequent imputation을 적용했다.

            연속형 변수는 scaling 전에 train-fold에서 계산한 IQR 범위로 clipping한다. 이렇게
            하면 극단값이 StandardScaler의 평균과 표준편차를 과도하게 흔드는 것을 줄이면서도
            새 입력을 동일한 pipeline으로 처리할 수 있다. 코드형 범주 변수는
            handle_unknown='ignore'를 둔 one-hot encoding을 사용해, 학습 fold에 없던 category가
            추론 시 들어와도 파이프라인이 깨지지 않게 했다. 중복 행과 target이 빈 행은 split
            전에 제거했다.
            """,
            FOOTER,
        )
        add_table_page(
            pdf,
            "MLflow 실험 기반 모델 비교",
            model_comparison_for_report(comparison),
            "표 1. 각 run은 parameter, 5-fold CV metric, test metric, 선택 특성, confusion matrix, 학습된 model artifact를 MLflow에 기록한다. 최종 모델은 최고 test balanced accuracy와 0.03 이내인 후보 중 recall, F1, balanced accuracy 순으로 선택했다.",
            FOOTER,
            font_size=8.0,
        )
        add_final_model_page(
            pdf,
            final_summary,
            ARTIFACTS_DIR / f"{final_run}_confusion_matrix.png",
            final_confusion_counts,
            FOOTER,
        )
        add_text_page(
            pdf,
            "테스트, 패키징, CI, Repo 구성",
            """
            unittest는 작지만 실제로 깨질 수 있는 지점을 겨냥했다. 예측 결과의 row 수가 입력
            row 수와 일치하는지, predict_proba 값이 [0, 1] 범위에 있고 행별 합이 1인지,
            cholesterol이 허용 임상 범위를 벗어나면 거부되는지, 같은 seed의 두 pipeline이 같은
            입력에 대해 같은 예측을 내는지를 확인한다.

            Dockerfile은 python:3.10-slim을 기반으로 pinned requirements를 설치하고, source,
            data, artifacts, final model을 복사한 뒤 data/sample_input.csv에 대한 batch inference를
            실행한다. GitHub Actions workflow는 push와 pull request마다 의존성을 설치하고
            python -m unittest를 실행한다.

            Repo에는 과제 체크리스트의 핵심 파일을 모두 포함했다: data, notebooks/01_eda_preprocessing.ipynb,
            src/preprocessing.py, src/train.py, src/inference.py, src/monitor.py, tests/test_pipeline.py,
            mlruns, Dockerfile, requirements.txt, .github/workflows/ci.yml, report.pdf, README.md. README는
            설치, 학습, 추론, 테스트, 모니터링, 보고서 재생성, Docker build 순서를 한 번에 따라갈
            수 있게 정리했다.

            feature store 후보로는 최대 심박수 변수 thalach가 적절하다. 학습, 추론, drift 점검에
            반복적으로 쓰이므로 schema와 freshness 검증 이점이 크다. model registry에는
            selected_features를 남기고 싶다. 어떤 변환 특성이 SelectFromModel 이후 살아남았는지
            보여 주기 때문에, 모델 버전 비교와 감사 가능성을 높인다.
            """,
            FOOTER,
        )
        add_table_page(
            pdf,
            "KS 검정 기반 데이터 드리프트",
            drift,
            "표 2. monitor.py는 test set 복사본에서 cholesterol과 oldpeak를 인위적으로 이동시킨 뒤, 각 연속형 변수에 scipy.stats.ks_2samp를 적용하고 p < 0.05를 drift로 표시한다. 이동시키지 않은 변수들이 flag되지 않는 것도 sanity check 역할을 한다.",
            FOOTER,
        )
        add_table_page(
            pdf,
            "원본 테스트셋과 드리프트 테스트셋 성능 비교",
            performance,
            "표 3. 합성 input drift가 곧바로 모델의 임상적 무효를 의미하지는 않는다. 다만 balanced accuracy 하락은 계속 사용하기 전 재검토가 필요하다는 명확한 신호다.",
            FOOTER,
        )
        add_image_page(
            pdf,
            "시간에 따른 모니터링 지표",
            ARTIFACTS_DIR / "drift_performance_timeseries.png",
            "그림 3. 합성 timestamp에 따라 cholesterol shift를 키웠을 때 balanced accuracy가 어떻게 변하는지 보여 준다. 운영 환경에서는 같은 방식으로 model version과 data window별 성능을 추적한다.",
            FOOTER,
        )
        add_text_page(
            pdf,
            "서빙, 재학습, 윤리, AI 사용 공개",
            """
            이 프로젝트는 on-device보다 Model-as-a-Service 형태로 서빙하는 편이 적절하다고
            판단했다. MaaS는 모델 업데이트, audit logging, monitoring, rollback을 단순하게 만든다.
            이 과제 수준에서는 작은 latency 이점보다 버전 관리와 감시 가능성이 더 중요하다. PHI는
            전송 암호화, 엄격한 접근 제어, 최소 보관, 로그 redaction으로 보호해야 한다. On-device는
            데이터 이동을 줄일 수 있지만, 모니터링과 긴급 업데이트가 어려워진다.

            재학습은 확인된 drift와 사람이 검토한 label이 함께 있을 때, labeled follow-up data에서
            성능 저하가 지속될 때, 또는 정기 임상 review 주기에 맞춰 수행한다. Human-in-the-loop
            검토는 새 label이 training data에 들어가기 전과 새 model version을 올리기 전에 필요하다.
            모델이 영향을 준 진료 결정이 그대로 미래 label에 섞이면 runaway feedback loop가
            생길 수 있기 때문이다.

            가장 큰 한계는 데이터셋이다. UCI Heart Disease는 작고 오래되었으며 현대의 다양한 환자
            집단을 충분히 대표하지 않는다. 시간이 한 주 더 있었다면 calibration 분석, subgroup별
            성능 점검, 더 명확한 data card, 그리고 임상의가 이해할 수 있는 위험 요인 설명 화면을
            추가했을 것이다.

            AI 사용 공개. AI 도구는 boilerplate 구현, debugging, 문서 초안 정리에 사용했다. 최종
            코드, 실험 선택, 결과 해석, 제출물에 대한 책임은 본인에게 있다.
            """,
            FOOTER,
        )

    return REPORT_PATH


if __name__ == "__main__":
    path = generate_report()
    print(f"Generated {path}")
