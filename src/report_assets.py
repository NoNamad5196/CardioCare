from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cardiocare")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "report_materials"
ZIP_PATH = OUTPUT_DIR / "CardioCare_report_materials.zip"
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def configure_fonts() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def clean_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for child in OUTPUT_DIR.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def save_table_png(df: pd.DataFrame, title: str, path: Path, font_size: float = 9.0) -> None:
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.3f}")

    fig_height = max(2.6, 0.45 * len(display) + 1.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#18314f", pad=14)
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e6eef8")
            cell.set_text_props(weight="bold", color="#18314f")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_eda_png() -> None:
    df = pd.read_csv(DATA_PATH, na_values=["?", ""])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.patch.set_facecolor("white")
    fig.suptitle("CardioCare EDA 요약", x=0.04, y=1.02, ha="left", fontsize=18, fontweight="bold", color="#18314f")

    target_counts = df["target"].value_counts(normalize=True).sort_index()
    axes[0].bar(["0: 정상", "1: 심장병"], target_counts.values, color=["#4c78a8", "#f58518"])
    axes[0].set_title("타깃 클래스 비율")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("비율")
    for idx, value in enumerate(target_counts.values):
        axes[0].text(idx, value + 0.025, f"{value:.1%}", ha="center", fontsize=10)

    missing_rates = df.isna().mean().sort_values(ascending=False)
    missing_rates = missing_rates[missing_rates > 0].head(6)
    axes[1].barh(missing_rates.index[::-1], missing_rates.values[::-1], color="#72b7b2")
    axes[1].set_title("결측률 상위 변수")
    axes[1].set_xlabel("결측률")
    axes[1].set_xlim(0, max(0.55, float(missing_rates.max()) * 1.2))
    for y_pos, value in enumerate(missing_rates.values[::-1]):
        axes[1].text(value + 0.01, y_pos, f"{value:.1%}", va="center", fontsize=10)

    outlier_counts = []
    for column in CONTINUOUS_FEATURES:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts.append(int(((series < lower) | (series > upper)).sum()))
    axes[2].bar(CONTINUOUS_FEATURES, outlier_counts, color="#54a24b")
    axes[2].set_title("IQR 이상치 개수")
    axes[2].tick_params(axis="x", rotation=30)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_eda_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_selected_features_file() -> None:
    metadata_path = MODELS_DIR / "final_model_metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata.get("selected_features", [])
    rows = ["# 최종 모델 선택 특성", ""]
    rows.extend(f"- {feature}" for feature in features)
    (OUTPUT_DIR / "selected_features.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def copy_existing_assets() -> None:
    copies = [
        ARTIFACTS_DIR / "model_comparison.csv",
        ARTIFACTS_DIR / "drift_report.csv",
        ARTIFACTS_DIR / "drift_performance_comparison.csv",
        ARTIFACTS_DIR / "performance_timeseries.csv",
        ARTIFACTS_DIR / "drift_performance_timeseries.png",
        ARTIFACTS_DIR / "predictions.csv",
        ARTIFACTS_DIR / "monitor_summary.json",
        MODELS_DIR / "final_model_metadata.json",
    ]
    metadata_path = MODELS_DIR / "final_model_metadata.json"
    if metadata_path.exists():
        selected_run = json.loads(metadata_path.read_text(encoding="utf-8")).get("selected_run")
        if selected_run:
            copies.extend(
                [
                    ARTIFACTS_DIR / f"{selected_run}_confusion_matrix.csv",
                    ARTIFACTS_DIR / f"{selected_run}_confusion_matrix.png",
                    ARTIFACTS_DIR / f"{selected_run}_selected_features.txt",
                ]
            )

    for source in copies:
        if source.exists():
            shutil.copy2(source, OUTPUT_DIR / source.name)


def make_table_images() -> None:
    comparison = pd.read_csv(ARTIFACTS_DIR / "model_comparison.csv")
    comparison_display = comparison.rename(
        columns={
            "run_name": "run",
            "model_family": "model",
            "selected_feature_count": "features",
            "cv_balanced_accuracy_mean": "cv_bAcc",
            "cv_precision_mean": "cv_prec",
            "cv_recall_mean": "cv_recall",
            "cv_f1_mean": "cv_F1",
            "test_balanced_accuracy": "test_bAcc",
            "test_precision": "test_prec",
            "test_recall": "test_recall",
            "test_f1": "test_F1",
        }
    )
    comparison_display["run"] = comparison_display["run"].replace(
        {
            "baseline_logistic_regression": "base_logreg",
            "baseline_random_forest": "base_rf",
            "baseline_svc": "base_svc",
            "tuned_svc": "tuned_svc",
        }
    )
    comparison_display["model"] = comparison_display["model"].replace(
        {
            "logistic_regression": "logreg",
            "random_forest": "rf",
        }
    )
    save_table_png(comparison_display, "MLflow 모델 비교표", OUTPUT_DIR / "02_model_comparison_table.png", font_size=8.5)

    drift = pd.read_csv(ARTIFACTS_DIR / "drift_report.csv")
    save_table_png(drift, "KS 검정 드리프트 결과", OUTPUT_DIR / "03_drift_report_table.png", font_size=9.5)

    performance = pd.read_csv(ARTIFACTS_DIR / "drift_performance_comparison.csv")
    save_table_png(performance, "원본 vs. 드리프트 성능 비교", OUTPUT_DIR / "04_drift_performance_table.png", font_size=10.0)


def write_readme() -> None:
    content = """# CardioCare 보고서 작성용 자료

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
"""
    (OUTPUT_DIR / "README_report_materials.md").write_text(content, encoding="utf-8")


def create_zip() -> Path:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.iterdir()):
            if path == ZIP_PATH:
                continue
            archive.write(path, arcname=path.name)
    return ZIP_PATH


def main() -> None:
    configure_fonts()
    clean_output_dir()
    make_eda_png()
    make_table_images()
    make_selected_features_file()
    copy_existing_assets()
    write_readme()
    zip_path = create_zip()
    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
