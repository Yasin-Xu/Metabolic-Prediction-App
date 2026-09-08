"""Train the fixed-feature Model A used by the Streamlit application.

The source workbook is read-only. Participant IDs are deduplicated with the
same deterministic rule used for Models E and F. All preprocessing, tuning,
threshold selection, and evaluation are restricted to the development split;
the deployable coefficients are then refit on all eligible participants with
the selected hyperparameter frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from train_cross_sectional_models import (
    ID_COLUMN,
    OUTCOME_COMPONENTS,
    RANDOM_STATE,
    TARGET,
    bootstrap_intervals,
    calibration_parameters,
    choose_threshold,
    coefficient_table,
    input_reference,
    make_pipeline,
    metric_values,
    select_regularization_strength,
    software_versions,
    to_builtin,
)


MODEL_FILE = "lab_model_a.pkl"
MODEL_DISPLAY_NAME = "模型A（体成分+实验室）"
RAW_SOURCE_SHEET = "导出数据2"
DEFAULT_RAW_SOURCE_NAME = "SS-导出初始数据集20240804.xlsx"

MODEL_FEATURES = [
    "体重指数",
    "收缩压",
    "下肢肌肉比率",
    "细胞外液总量/身体总水分",
    "上肢肌肉比率",
    "下肢脂肪百分比",
    "糖化血红蛋白",
    "总胆固醇",
    "甘油三酯",
    "TyG 指数",
]

SOURCE_COLUMNS = {
    "体重指数": "体重指数",
    "收缩压": "收缩压",
    "下肢肌肉比率": "下肢肌肉比率",
    "细胞外液总量/身体总水分": "细胞外液总量/身体总水分",
    "上肢肌肉比率": "上肢肌肉比率",
    "下肢脂肪百分比": "下肢脂肪百分比",
    "糖化血红蛋白": "糖化血红蛋白",
    "总胆固醇": "总胆固醇",
    "甘油三酯": "甘油三酯",
}

GLUCOSE_SOURCE_COLUMN = "葡萄糖"
TRIGLYCERIDE_MMOL_L_TO_MG_DL = 88.5
GLUCOSE_MMOL_L_TO_MG_DL = 18.0
RAW_MISSINGNESS_COLUMNS = [
    "糖化血红蛋白",
    "总胆固醇",
    "甘油三酯",
    GLUCOSE_SOURCE_COLUMN,
    "收缩压",
]


def calculate_tyg(triglyceride_mmol_l: Any, glucose_mmol_l: Any):
    """Return TyG = ln[TG(mg/dL) * fasting glucose(mg/dL) / 2]."""
    triglyceride = np.asarray(triglyceride_mmol_l, dtype=float)
    glucose = np.asarray(glucose_mmol_l, dtype=float)
    product = (
        triglyceride
        * TRIGLYCERIDE_MMOL_L_TO_MG_DL
        * glucose
        * GLUCOSE_MMOL_L_TO_MG_DL
        / 2.0
    )
    result = np.where(product > 0, np.log(product), np.nan)
    if result.ndim == 0:
        return float(result)
    return result


def restore_original_missingness(
    current: pd.DataFrame,
    raw_source_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restore only the missing-value mask from the pre-imputation export.

    The initial export is keyed uniquely by participant ID, while the current
    workbook can contain repeated IDs. The mapping is therefore deliberately
    many-to-one. Reference values are used only to verify that the correct
    source has been supplied; they are never copied into the current data.
    """
    if not raw_source_path.is_file():
        raise FileNotFoundError(
            f"找不到原始缺失来源文件: {raw_source_path.name}。"
            "请通过 --raw-source 指定原始导出工作簿。"
        )

    reference = pd.read_excel(raw_source_path, sheet_name=RAW_SOURCE_SHEET)
    required = {ID_COLUMN, *RAW_MISSINGNESS_COLUMNS}
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise ValueError(f"原始缺失来源工作簿缺少必要字段: {missing}")
    if reference[ID_COLUMN].isna().any():
        raise ValueError("原始缺失来源工作簿的体检号存在缺失。")
    if reference[ID_COLUMN].duplicated().any():
        duplicate_ids = reference.loc[
            reference[ID_COLUMN].duplicated(keep=False), ID_COLUMN
        ].drop_duplicates()
        raise ValueError(
            "原始缺失来源工作簿的体检号不是唯一键；示例: "
            f"{duplicate_ids.head(10).tolist()}"
        )
    if current[ID_COLUMN].isna().any():
        raise ValueError("当前源工作簿的体检号存在缺失，无法恢复原始缺失掩码。")

    reference = reference.set_index(ID_COLUMN, verify_integrity=True)
    current_ids = pd.Index(current[ID_COLUMN].unique())
    unmatched_ids = current_ids.difference(reference.index)
    if len(unmatched_ids):
        raise ValueError(
            "当前源工作簿存在无法映射到原始导出的体检号；示例: "
            f"{unmatched_ids[:10].tolist()}"
        )

    restored = current.copy()
    union_missing_mask = pd.Series(False, index=restored.index)
    field_audit: dict[str, Any] = {}
    for field in RAW_MISSINGNESS_COLUMNS:
        before = restored[field].copy()
        current_numeric = pd.to_numeric(before, errors="coerce")
        reference_numeric = pd.to_numeric(
            restored[ID_COLUMN].map(reference[field]), errors="coerce"
        )
        reference_nonmissing = reference_numeric.notna()
        comparable = reference_nonmissing & current_numeric.notna()
        equal = pd.Series(False, index=restored.index)
        equal.loc[comparable] = np.isclose(
            current_numeric.loc[comparable],
            reference_numeric.loc[comparable],
            rtol=0.0,
            atol=1e-12,
        )
        mismatch = reference_nonmissing & ~equal
        if mismatch.any():
            examples = restored.loc[mismatch, ID_COLUMN].head(10).tolist()
            raise ValueError(
                f"字段‘{field}’在原始非缺失记录上与当前数据不一致，"
                f"无法安全应用缺失掩码；不一致 {int(mismatch.sum())} 行，"
                f"示例体检号: {examples}"
            )

        reference_missing = reference_numeric.isna()
        union_missing_mask |= reference_missing
        values_set_missing = reference_missing & current_numeric.notna()
        restored.loc[reference_missing, field] = np.nan

        # Guard against accidental replacement of any observed source value.
        # Setting NaN can promote an integer column to float, so compare the
        # numeric values rather than requiring pandas dtype equality.
        preserved = ~reference_missing
        after_numeric = pd.to_numeric(restored[field], errors="coerce")
        if not np.allclose(
            after_numeric.loc[preserved].to_numpy(dtype=float),
            current_numeric.loc[preserved].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise AssertionError(f"字段‘{field}’的非缺失值被意外改写。")

        field_audit[field] = {
            "reference_missing_rows": int(reference_missing.sum()),
            "reference_missing_unique_participants": int(
                restored.loc[reference_missing, ID_COLUMN].nunique()
            ),
            "current_values_restored_to_missing": int(values_set_missing.sum()),
            "reference_nonmissing_rows_verified": int(reference_nonmissing.sum()),
            "exact_matches_on_reference_nonmissing": int(equal.sum()),
        }

    audit = {
        # Store only portable provenance, never the user's absolute path.
        "raw_source_file_name": raw_source_path.name,
        "raw_source_sheet": RAW_SOURCE_SHEET,
        "raw_source_sha256": hashlib.sha256(raw_source_path.read_bytes()).hexdigest(),
        "raw_source_rows": int(len(reference)),
        "raw_source_unique_participants": int(reference.index.nunique()),
        "mapped_current_rows": int(len(restored)),
        "mapped_current_unique_participants": int(current_ids.nunique()),
        "unmatched_current_participants": 0,
        "fields": field_audit,
        "any_original_missing_rows": int(union_missing_mask.sum()),
        "any_original_missing_unique_participants": int(
            restored.loc[union_missing_mask, ID_COLUMN].nunique()
        ),
    }
    return restored, audit


def prepare_model_data(
    source_path: Path,
    raw_source_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    raw = pd.read_excel(source_path, sheet_name="Sheet1")
    required = {
        ID_COLUMN,
        TARGET,
        "身体质量指数",
        GLUCOSE_SOURCE_COLUMN,
        *OUTCOME_COMPONENTS,
        *SOURCE_COLUMNS.values(),
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"源工作簿缺少必要字段: {missing}")

    raw, missingness_audit = restore_original_missingness(raw, raw_source_path)
    raw["_source_row"] = np.arange(len(raw)) + 2
    raw["_bmi_consistency_gap"] = (
        pd.to_numeric(raw["身体质量指数"], errors="coerce")
        - pd.to_numeric(raw["体重指数"], errors="coerce")
    ).abs()
    duplicate_rows = int(raw[ID_COLUMN].duplicated(keep=False).sum())
    duplicate_ids = int(raw.loc[raw[ID_COLUMN].duplicated(keep=False), ID_COLUMN].nunique())
    deduplicated = (
        raw.sort_values([ID_COLUMN, "_bmi_consistency_gap", "_source_row"], na_position="last")
        .drop_duplicates(ID_COLUMN, keep="first")
        .sort_values("_source_row")
        .reset_index(drop=True)
    )
    missingness_audit["post_dedup_missing_counts"] = {
        field: int(pd.to_numeric(deduplicated[field], errors="coerce").isna().sum())
        for field in RAW_MISSINGNESS_COLUMNS
    }

    analysis = pd.DataFrame(index=deduplicated.index)
    analysis[ID_COLUMN] = deduplicated[ID_COLUMN]
    analysis[TARGET] = pd.to_numeric(deduplicated[TARGET], errors="raise").astype(int)
    for canonical, source in SOURCE_COLUMNS.items():
        analysis[canonical] = pd.to_numeric(deduplicated[source], errors="coerce")
    glucose = pd.to_numeric(deduplicated[GLUCOSE_SOURCE_COLUMN], errors="coerce")

    # The source contains a distinct block of impossible systolic values (0–46
    # mmHg) and one HbA1c value of zero. Treat these as unavailable and let the
    # fold-fitted pipeline impute them instead of learning from data errors.
    invalid_sbp = analysis["收缩压"].notna() & ~analysis["收缩压"].between(80, 250)
    invalid_hba1c = analysis["糖化血红蛋白"].notna() & ~analysis["糖化血红蛋白"].between(3, 20)
    invalid_bmi = analysis["体重指数"].notna() & ~analysis["体重指数"].between(10, 70)
    invalid_ratio = pd.Series(False, index=analysis.index)
    for name in ["下肢肌肉比率", "细胞外液总量/身体总水分", "上肢肌肉比率", "下肢脂肪百分比"]:
        current = analysis[name].notna() & ~analysis[name].between(0, 1)
        invalid_ratio |= current
        analysis.loc[current, name] = np.nan
    analysis.loc[invalid_sbp, "收缩压"] = np.nan
    analysis.loc[invalid_hba1c, "糖化血红蛋白"] = np.nan
    analysis.loc[invalid_bmi, "体重指数"] = np.nan

    invalid_tyg_source = (
        glucose.isna()
        | glucose.le(0)
        | analysis["甘油三酯"].isna()
        | analysis["甘油三酯"].le(0)
    )
    analysis["TyG 指数"] = calculate_tyg(analysis["甘油三酯"], glucose)
    analysis.loc[invalid_tyg_source, "TyG 指数"] = np.nan

    if not analysis[TARGET].isin([0, 1]).all():
        raise ValueError("因变量‘代谢异常’必须仅包含0/1。")
    outcome_components = deduplicated[OUTCOME_COMPONENTS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if outcome_components.isna().any().any():
        raise ValueError("因变量组成字段存在缺失，无法核对‘代谢异常’标签。")
    reconstructed_target = outcome_components.sum(axis=1).ge(2).astype(int)
    outcome_rule_matches = reconstructed_target.eq(analysis[TARGET])
    triglyceride_tyg_spearman = float(
        analysis[["甘油三酯", "TyG 指数"]].corr(method="spearman").iloc[0, 1]
    )

    audit = {
        "source_rows": int(len(raw)),
        "source_unique_participants": int(raw[ID_COLUMN].nunique()),
        "source_sha256": source_sha256,
        "raw_missingness_restoration": missingness_audit,
        "duplicate_rows": duplicate_rows,
        "duplicate_participant_ids": duplicate_ids,
        "analysis_rows": int(len(analysis)),
        "outcome_positive": int(analysis[TARGET].sum()),
        "outcome_negative": int((analysis[TARGET] == 0).sum()),
        "outcome_prevalence": float(analysis[TARGET].mean()),
        "outcome_component_columns": OUTCOME_COMPONENTS,
        "outcome_rule": "four stored component indicators sum >= 2",
        "outcome_rule_matches": int(outcome_rule_matches.sum()),
        "outcome_rule_all_rows_match": bool(outcome_rule_matches.all()),
        "invalid_sbp_set_missing": int(invalid_sbp.sum()),
        "invalid_hba1c_set_missing": int(invalid_hba1c.sum()),
        "invalid_bmi_set_missing": int(invalid_bmi.sum()),
        "invalid_ratio_values_set_missing": int(invalid_ratio.sum()),
        "invalid_tyg_source_set_missing": int(invalid_tyg_source.sum()),
        "triglyceride_tyg_spearman": triglyceride_tyg_spearman,
        "feature_missing_counts": {
            feature: int(analysis[feature].isna().sum()) for feature in MODEL_FEATURES
        },
        "tyg_formula": "ln[(TG mmol/L × 88.5) × (fasting glucose mmol/L × 18) / 2]",
        "tyg_reference": {
            "minimum": float(analysis["TyG 指数"].min()),
            "median": float(analysis["TyG 指数"].median()),
            "maximum": float(analysis["TyG 指数"].max()),
        },
        "glucose_reference": {
            "minimum": float(glucose.min()),
            "percentile_1": float(glucose.quantile(0.01)),
            "median": float(glucose.median()),
            "percentile_99": float(glucose.quantile(0.99)),
            "maximum": float(glucose.max()),
        },
    }
    return analysis, audit


def train_model(
    source_path: Path,
    raw_source_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    analysis, audit = prepare_model_data(source_path, raw_source_path)
    x = analysis[MODEL_FEATURES]
    y = analysis[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    selected_c, tuning = select_regularization_strength(
        x_train,
        y_train,
        MODEL_FEATURES,
        cv,
    )
    selected_pipeline = make_pipeline(MODEL_FEATURES, selected_c)
    oof_probabilities = cross_val_predict(
        clone(selected_pipeline),
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold = choose_threshold(y_train, oof_probabilities)

    evaluation_model = clone(selected_pipeline).fit(x_train, y_train)
    test_probabilities = evaluation_model.predict_proba(x_test)[:, 1]
    metrics = metric_values(y_test, test_probabilities, threshold)
    metrics.update(calibration_parameters(y_test, test_probabilities))
    intervals = bootstrap_intervals(y_test, test_probabilities, threshold)

    deploy_model = clone(selected_pipeline).fit(x, y)
    metadata = {
        "display_name": MODEL_DISPLAY_NAME,
        "development_date": date.today().isoformat(),
        "software_versions": software_versions(),
        "source_sha256": audit["source_sha256"],
        "raw_missingness_source": {
            "file_name": audit["raw_missingness_restoration"]["raw_source_file_name"],
            "sheet": audit["raw_missingness_restoration"]["raw_source_sheet"],
            "sha256": audit["raw_missingness_restoration"]["raw_source_sha256"],
            "mapping_key": ID_COLUMN,
            "fields": RAW_MISSINGNESS_COLUMNS,
            "policy": (
                "Restore only the original missing-value mask before "
                "deduplication and TyG derivation; never overwrite observed values."
            ),
        },
        "outcome": TARGET,
        "positive_class": 1,
        "candidate_features": MODEL_FEATURES,
        "derived_input": {
            "feature": "TyG 指数",
            "requires": ["甘油三酯", "葡萄糖"],
            "formula": audit["tyg_formula"],
            "auxiliary_input_reference": audit["glucose_reference"],
        },
        "regularization": {
            "penalty": "L1 (LASSO)",
            "solver": "liblinear",
            "selected_c": selected_c,
        },
        "classification_threshold": threshold,
        "threshold_method": "maximum Youden J on 10-fold out-of-fold development predictions",
        "evaluation": {
            "design": "fixed stratified 80/20 hold-out; preprocessing, tuning, and threshold selection restricted to development data",
            "random_state": RANDOM_STATE,
            "train_n": int(len(x_train)),
            "test_n": int(len(x_test)),
            "test_positive_n": int(y_test.sum()),
            "test_prevalence": float(y_test.mean()),
            "metrics": metrics,
            "bootstrap_95_intervals": intervals,
        },
        "selected_terms_in_full_refit": coefficient_table(deploy_model),
        "input_reference": input_reference(analysis, MODEL_FEATURES),
        "data_quality": audit,
        "interpretation_warning": (
            "Several specified predictors are measured at the same examination and overlap with "
            "the recorded outcome components. Performance therefore describes same-dataset "
            "classification and must not be interpreted as prospective incidence prediction. "
            "Triglyceride and TyG are strongly correlated, so their conditional coefficients "
            "must not be interpreted as independent or causal effects."
        ),
        "deployment_refit": "Hyperparameter and threshold frozen, then coefficients refit on all eligible participants.",
    }
    bundle = {"model": deploy_model, "threshold": threshold, "metadata": metadata}
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / MODEL_FILE, compress=3)

    report = {
        "data_audit": audit,
        "model_file": MODEL_FILE,
        "metadata": metadata,
        "tuning": tuning,
    }
    (output_dir / "lab_model_a_report.json").write_text(
        json.dumps(to_builtin(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    audit = report["data_audit"]
    restoration = audit["raw_missingness_restoration"]
    metadata = report["metadata"]
    metrics = metadata["evaluation"]["metrics"]
    lines = [
        "# 模型A开发记录",
        "",
        "## 固定输入指标",
        "",
        *[f"- {feature}" for feature in MODEL_FEATURES],
        "",
        "TyG指数按 `ln[(TG mmol/L × 88.5) × (空腹血糖 mmol/L × 18) / 2]` 计算；该换算口径与原项目TyG数据逐行一致。",
        "",
        "## 数据与处理",
        "",
        f"- 源数据 {audit['source_rows']} 行、{audit['source_unique_participants']} 名唯一受试者。",
        (
            "- 原始缺失掩码来自 `"
            f"{restoration['raw_source_file_name']}` 的 "
            f"`{restoration['raw_source_sheet']}` 工作表，按 `{ID_COLUMN}` "
            "many-to-one 映射；不复制或覆盖任何原始非缺失数值。"
        ),
        (
            "- 恢复原始缺失："
            + "、".join(
                f"{field} {details['current_values_restored_to_missing']} 条"
                for field, details in restoration["fields"].items()
            )
            + f"；并集 {restoration['any_original_missing_rows']} 条。"
        ),
        f"- 去重后 {audit['analysis_rows']} 名；阳性 {audit['outcome_positive']} 名（{audit['outcome_prevalence']:.1%}）。",
        (
            "- 当前‘代谢异常’列与工作簿内高血压/血压异常、糖尿病/糖代谢异常、"
            f"甘油三脂异常及HDL异常四项合计≥2完全一致：{audit['outcome_rule_matches']} / "
            f"{audit['analysis_rows']} 条。"
        ),
        f"- 非生理收缩压设为缺失并在折内插补：{audit['invalid_sbp_set_missing']} 条。",
        f"- 非生理糖化血红蛋白设为缺失并在折内插补：{audit['invalid_hba1c_set_missing']} 条。",
        f"- 甘油三酯与TyG的Spearman相关系数为 {audit['triglyceride_tyg_spearman']:.3f}；二者同时入模时，单项系数仅作条件关联，不作独立或因果解释。",
        "- 训练、调参与阈值选择仅在80%开发集完成；20%测试集仅用于锁定后评估。",
        "",
        "## 内部测试集表现",
        "",
        f"- ROC AUC：{metrics['roc_auc']:.3f}",
        f"- PR AUC：{metrics['average_precision']:.3f}",
        f"- Brier score：{metrics['brier_score']:.3f}",
        f"- 灵敏度：{metrics['sensitivity']:.3f}",
        f"- 特异度：{metrics['specificity']:.3f}",
        f"- 固定判别阈值：{metadata['classification_threshold']:.4f}",
        "",
        "## 解释限制",
        "",
        "用户指定的糖化血红蛋白、总胆固醇、甘油三酯和TyG均来自同次检查，且部分与记录的结局组成存在直接或间接重叠。内部性能可能因此偏高，不应解释为未来发病预测；正式应用前需要独立数据验证。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source xlsx workbook")
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=None,
        help=(
            "Pre-imputation source workbook used only to restore the original "
            f"missing-value mask. Defaults to {DEFAULT_RAW_SOURCE_NAME!r} next "
            "to the source workbook."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    raw_source_path = (
        args.raw_source.resolve()
        if args.raw_source is not None
        else source_path.with_name(DEFAULT_RAW_SOURCE_NAME)
    )
    report = train_model(
        source_path,
        raw_source_path,
        args.output_dir.resolve(),
    )
    (args.output_dir / "LAB_MODEL_A_REPORT.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps({
        "model_file": str((args.output_dir / MODEL_FILE).resolve()),
        "threshold": report["metadata"]["classification_threshold"],
        "test_metrics": report["metadata"]["evaluation"]["metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
