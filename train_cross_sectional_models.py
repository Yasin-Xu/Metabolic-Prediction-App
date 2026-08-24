"""Train and document the two cross-sectional LASSO logistic models.

The script keeps the source workbook read-only. It creates a processed analysis
table in memory, performs an untouched stratified hold-out evaluation, and then
refits the deployable models on all eligible participants with the selected
regularization strength frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 20260824
TARGET = "代谢异常"
ID_COLUMN = "体检号"
OUTCOME_COMPONENTS = [
    "高血压/血压异常",
    "糖尿病/糖代谢异常",
    "甘油三脂异常",
    "HDL异常",
]

SOURCE_COLUMN_MAP = {
    "性别": "性别（1男，0女）",
    "年龄": "年龄",
    "吸烟史": "吸烟史（0，不吸；1，吸烟）",
    "饮酒史": "饮酒史（0，否认，1饮酒）",
    "体重指数": "体重指数",
    "腰臀比": "腰臀比",
    "身体总水分": "身体总水分",
    "体脂肪": "体脂肪",
    "体脂百分比": "体脂百分比",
    "上肢肌肉比率": "上肢肌肉比率",
    "躯干肌肉量比率": "躯干肌肉量比率",
    "下肢肌肉比率": "下肢肌肉比率",
    "细胞外液总量/身体总水分": "细胞外液总量/身体总水分",
    "上肢脂肪百分比": "上肢脂肪百分比",
    "躯干脂肪百分比": "躯干脂肪百分比",
    "下肢脂肪百分比": "下肢脂肪百分比",
    "身体总水分/去脂体重": "身体总水分/去脂体重",
}

DERIVED_SOURCE_COLUMNS = {
    "运动频率": "运动频率（0，每周<1词；6，每周＞5词）",
    "上肢细胞外液总量/身体总水分": {
        "numerator": ["右上肢细胞外液总量", "左上肢细胞外液总量"],
        "denominator": ["右上肢总水分", "左上肢总水分"],
    },
}

BASIC_FEATURES = [
    "性别",
    "年龄",
    "运动频率",
    "吸烟史",
    "饮酒史",
    "体重指数",
    "腰臀比",
]

BODY_COMPOSITION_FEATURES = BASIC_FEATURES + [
    "身体总水分",
    "体脂肪",
    "体脂百分比",
    "上肢肌肉比率",
    "躯干肌肉量比率",
    "下肢肌肉比率",
    "细胞外液总量/身体总水分",
    "上肢细胞外液总量/身体总水分",
    "上肢脂肪百分比",
    "躯干脂肪百分比",
    "下肢脂肪百分比",
    "身体总水分/去脂体重",
]

CATEGORICAL_FEATURES = ["性别", "运动频率", "吸烟史", "饮酒史"]

MODEL_SPECS = {
    "cross_sectional_bodycomp_lasso": {
        "display_name": "横断面模型E（人口学+体成分）",
        "features": BODY_COMPOSITION_FEATURES,
        "filename": "cross_sectional_bodycomp_lasso.pkl",
    },
    "cross_sectional_basic_lasso": {
        "display_name": "横断面模型F（人口学基础）",
        "features": BASIC_FEATURES,
        "filename": "cross_sectional_basic_lasso.pkl",
    },
}

REPORT_FIELD_MAPPING = [
    ("性别", "性别（1男，0女）"),
    ("年龄", "年龄"),
    ("运动频率", "运动频率（0…6）→ <1、1–2、3–5、>5 次/周四组"),
    ("吸烟史", "吸烟史（0，不吸；1，吸烟）（二分类，含戒烟者为有史）"),
    ("饮酒史", "饮酒史（0，否认，1饮酒）（二分类）"),
    ("BMI", "体重指数（临床身高/体重计算；未使用 InBody‘身体质量指数’）"),
    ("WHR", "腰臀比（腰围/臀围；未使用‘腰臀比INBODY’）"),
    ("身体总水分", "身体总水分"),
    ("体脂肪量", "体脂肪"),
    ("体脂百分比", "体脂百分比（百分数点）"),
    ("上肢肌肉比率", "上肢肌肉比率"),
    ("躯干肌肉比率", "躯干肌肉量比率"),
    ("下肢肌肉比率", "下肢肌肉比率"),
    ("ECW/TBW", "细胞外液总量/身体总水分"),
    (
        "上肢 ECW/TBW",
        "（右上肢细胞外液+左上肢细胞外液）/（右上肢总水分+左上肢总水分）",
    ),
    ("上肢脂肪比率", "上肢脂肪百分比（实际存储为0–1占比）"),
    ("躯干脂肪比率", "躯干脂肪百分比（实际存储为0–1占比）"),
    ("下肢脂肪比率", "下肢脂肪百分比（实际存储为0–1占比）"),
    ("TBW/FFM", "身体总水分/去脂体重（表中为百分数点；网页0–1输入后×100）"),
]


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def to_builtin(value: Any) -> Any:
    """Recursively convert numpy/pandas values to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def recode_exercise_frequency(series: pd.Series) -> pd.Series:
    """Map raw weekly frequency (0..6, including decimals) to the app's 4 groups."""
    values = pd.to_numeric(series, errors="coerce")
    groups = pd.cut(
        values,
        bins=[-np.inf, 0, 2, 5, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
        right=True,
    )
    return groups.astype("float64")


def prepare_analysis_data(source_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    raw = pd.read_excel(source_path, sheet_name="Sheet1")
    arm_ecw_sources = DERIVED_SOURCE_COLUMNS["上肢细胞外液总量/身体总水分"]
    required = {
        ID_COLUMN,
        TARGET,
        "身体质量指数",
        "高密度脂蛋白胆固醇",
        *OUTCOME_COMPONENTS,
        *SOURCE_COLUMN_MAP.values(),
        DERIVED_SOURCE_COLUMNS["运动频率"],
        *arm_ecw_sources["numerator"],
        *arm_ecw_sources["denominator"],
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"源工作簿缺少必要字段: {missing}")

    raw = raw.copy()
    raw["_source_row"] = np.arange(len(raw)) + 2
    raw["_bmi_consistency_gap"] = (
        pd.to_numeric(raw["身体质量指数"], errors="coerce")
        - pd.to_numeric(raw["体重指数"], errors="coerce")
    ).abs()

    duplicate_rows = int(raw[ID_COLUMN].duplicated(keep=False).sum())
    duplicate_ids = int(raw.loc[raw[ID_COLUMN].duplicated(keep=False), ID_COLUMN].nunique())

    # The workbook contains 16 participant IDs matched to two body-composition
    # records. Retain the record whose InBody BMI is most consistent with the
    # independently measured clinical BMI; source-row order resolves exact ties.
    deduplicated = (
        raw.sort_values([ID_COLUMN, "_bmi_consistency_gap", "_source_row"], na_position="last")
        .drop_duplicates(ID_COLUMN, keep="first")
        .sort_values("_source_row")
        .reset_index(drop=True)
    )

    analysis = pd.DataFrame(index=deduplicated.index)
    analysis[ID_COLUMN] = deduplicated[ID_COLUMN]
    analysis[TARGET] = pd.to_numeric(deduplicated[TARGET], errors="raise").astype(int)
    for canonical, source in SOURCE_COLUMN_MAP.items():
        analysis[canonical] = pd.to_numeric(deduplicated[source], errors="coerce")

    analysis["运动频率"] = recode_exercise_frequency(
        deduplicated[DERIVED_SOURCE_COLUMNS["运动频率"]]
    )
    arm_ecw_numerator = deduplicated[arm_ecw_sources["numerator"]].apply(
        pd.to_numeric, errors="coerce"
    ).sum(axis=1, min_count=1)
    arm_ecw_denominator = deduplicated[arm_ecw_sources["denominator"]].apply(
        pd.to_numeric, errors="coerce"
    ).sum(axis=1, min_count=1)
    analysis["上肢细胞外液总量/身体总水分"] = arm_ecw_numerator / arm_ecw_denominator.replace(0, np.nan)

    invalid_whr = analysis["腰臀比"].notna() & ~analysis["腰臀比"].between(0.5, 1.5)
    invalid_whr_count = int(invalid_whr.sum())
    analysis.loc[invalid_whr, "腰臀比"] = np.nan

    if not analysis[TARGET].isin([0, 1]).all():
        raise ValueError("因变量“代谢异常”必须仅包含 0/1。")

    component_sum = deduplicated[OUTCOME_COMPONENTS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    outcome_rule_matches = (component_sum.ge(2).astype(int) == analysis[TARGET])
    hdl = pd.to_numeric(deduplicated["高密度脂蛋白胆固醇"], errors="coerce")
    sex = pd.to_numeric(deduplicated[SOURCE_COLUMN_MAP["性别"]], errors="coerce")
    recalculated_hdl_abnormal = (
        (sex.eq(1) & hdl.lt(1.0)) | (sex.eq(0) & hdl.lt(1.3))
    ).astype(int)
    stored_hdl_abnormal = pd.to_numeric(deduplicated["HDL异常"], errors="coerce").astype(int)
    hdl_label_disagreement = int((stored_hdl_abnormal != recalculated_hdl_abnormal).sum())
    recalculated_component_sum = (
        deduplicated[["高血压/血压异常", "糖尿病/糖代谢异常", "甘油三脂异常"]]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
        + recalculated_hdl_abnormal
    )
    recalculated_target = recalculated_component_sum.ge(2).astype(int)
    target_changes_if_hdl_recalculated = int((recalculated_target != analysis[TARGET]).sum())
    raw_exercise = pd.to_numeric(raw[DERIVED_SOURCE_COLUMNS["运动频率"]], errors="coerce")
    raw_exercise_noninteger = int(
        (raw_exercise.notna() & ~np.isclose(raw_exercise, np.rint(raw_exercise))).sum()
    )

    audit = {
        "source_rows": int(len(raw)),
        "source_sha256": source_sha256,
        "source_unique_participants": int(raw[ID_COLUMN].nunique()),
        "duplicate_rows": duplicate_rows,
        "duplicate_participant_ids": duplicate_ids,
        "analysis_rows": int(len(analysis)),
        "invalid_whr_values_set_missing": invalid_whr_count,
        "outcome_positive": int(analysis[TARGET].sum()),
        "outcome_negative": int((analysis[TARGET] == 0).sum()),
        "outcome_prevalence": float(analysis[TARGET].mean()),
        "outcome_component_columns": OUTCOME_COMPONENTS,
        "outcome_rule": "four component indicators sum >= 2",
        "outcome_rule_matches": int(outcome_rule_matches.sum()),
        "outcome_rule_all_rows_match": bool(outcome_rule_matches.all()),
        "hdl_label_disagreement": hdl_label_disagreement,
        "target_changes_if_hdl_recalculated": target_changes_if_hdl_recalculated,
        "raw_exercise_noninteger_values": raw_exercise_noninteger,
        "participants_younger_than_18": int((analysis["年龄"] < 18).sum()),
        "participants_older_than_80": int((analysis["年龄"] > 80).sum()),
        "feature_missing_counts": {
            feature: int(analysis[feature].isna().sum())
            for feature in BODY_COMPOSITION_FEATURES
        },
    }
    return analysis, audit


def make_pipeline(features: list[str], c_value: float) -> Pipeline:
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    numeric = [feature for feature in features if feature not in categorical]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        verbose_feature_names_out=False,
    )
    classifier = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=float(c_value),
        max_iter=10_000,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def select_regularization_strength(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
    cv: StratifiedKFold,
) -> tuple[float, dict[str, Any]]:
    c_grid = np.logspace(-3, 2, 60)
    search = GridSearchCV(
        estimator=make_pipeline(features, c_value=1.0),
        param_grid={"classifier__C": c_grid},
        scoring="neg_log_loss",
        cv=cv,
        refit=False,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(x_train, y_train)

    mean_scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    std_scores = np.asarray(search.cv_results_["std_test_score"], dtype=float)
    standard_errors = std_scores / np.sqrt(cv.get_n_splits())
    best_index = int(np.nanargmax(mean_scores))
    one_se_floor = float(mean_scores[best_index] - standard_errors[best_index])
    eligible = np.flatnonzero(mean_scores >= one_se_floor)
    selected_index = int(eligible[0])  # smallest C = strongest penalty

    diagnostics = {
        "selection_rule": "10-fold cross-validated negative log loss (binomial deviance) with one-standard-error rule",
        "c_grid": c_grid,
        "mean_cv_negative_log_loss": mean_scores,
        "standard_error_cv_negative_log_loss": standard_errors,
        "best_mean_score_c": float(c_grid[best_index]),
        "best_mean_negative_log_loss": float(mean_scores[best_index]),
        "one_se_score_floor": one_se_floor,
        "selected_c": float(c_grid[selected_index]),
        "selected_mean_negative_log_loss": float(mean_scores[selected_index]),
    }
    return float(c_grid[selected_index]), diagnostics


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    youden_j = true_positive_rate - false_positive_rate
    valid_indices = np.flatnonzero(finite)
    best = valid_indices[int(np.argmax(youden_j[finite]))]
    return float(np.clip(thresholds[best], 0.0, 1.0))


def metric_values(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    negative_predictive_value = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "sensitivity": recall_score(y_true, predictions, zero_division=0),
        "specificity": specificity,
        "positive_predictive_value": precision_score(y_true, predictions, zero_division=0),
        "negative_predictive_value": negative_predictive_value,
        "f1": f1_score(y_true, predictions, zero_division=0),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
    }


def calibration_parameters(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000)
    calibration_model.fit(log_odds, np.asarray(y_true, dtype=int))
    return {
        "calibration_intercept": float(calibration_model.intercept_[0]),
        "calibration_slope": float(calibration_model.coef_[0, 0]),
    }


def bootstrap_intervals(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    iterations: int = 2_000,
) -> dict[str, dict[str, float]]:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    wanted = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "sensitivity",
        "specificity",
    ]
    samples = {name: [] for name in wanted}
    for _ in range(iterations):
        indices = rng.integers(0, len(y), len(y))
        sampled_y = y[indices]
        if np.unique(sampled_y).size < 2:
            continue
        values = metric_values(sampled_y, probabilities[indices], threshold)
        for name in wanted:
            samples[name].append(values[name])

    intervals = {}
    for name, values in samples.items():
        lower, upper = np.nanpercentile(values, [2.5, 97.5])
        intervals[name] = {"lower_95": float(lower), "upper_95": float(upper)}
    return intervals


def coefficient_table(model: Pipeline) -> list[dict[str, Any]]:
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    rows = [
        {
            "transformed_feature": str(name),
            "standardized_coefficient": float(coefficient),
            "odds_ratio_per_standard_deviation": float(np.exp(coefficient)),
        }
        for name, coefficient in zip(names, coefficients, strict=True)
        if not np.isclose(coefficient, 0.0, atol=1e-10)
    ]
    return sorted(rows, key=lambda row: abs(row["standardized_coefficient"]), reverse=True)


def input_reference(analysis: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    reference = {}
    for feature in features:
        series = analysis[feature]
        if feature in CATEGORICAL_FEATURES:
            reference[feature] = {
                "type": "categorical",
                "counts": {
                    str(int(level)): int(count)
                    for level, count in series.value_counts(dropna=True).sort_index().items()
                },
            }
        else:
            reference[feature] = {
                "type": "numeric",
                "median": float(series.median()),
                "percentile_1": float(series.quantile(0.01)),
                "percentile_99": float(series.quantile(0.99)),
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "missing": int(series.isna().sum()),
            }
    return reference


def train_one_model(
    analysis: pd.DataFrame,
    spec: dict[str, Any],
    output_dir: Path,
    data_audit: dict[str, Any],
) -> dict[str, Any]:
    features = list(spec["features"])
    x = analysis[features]
    y = analysis[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    selected_c, tuning = select_regularization_strength(x_train, y_train, features, cv)
    selected_pipeline = make_pipeline(features, selected_c)

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
    coefficients = coefficient_table(deploy_model)
    metadata = {
        "display_name": spec["display_name"],
        "development_date": date.today().isoformat(),
        "software_versions": software_versions(),
        "source_sha256": data_audit["source_sha256"],
        "study_design": "cross-sectional",
        "outcome": TARGET,
        "positive_class": 1,
        "candidate_features": features,
        "categorical_features": [f for f in features if f in CATEGORICAL_FEATURES],
        "regularization": {
            "penalty": "L1 (LASSO)",
            "solver": "liblinear",
            "selected_c": selected_c,
        },
        "classification_threshold": threshold,
        "threshold_method": "maximum Youden J on 10-fold out-of-fold training predictions",
        "evaluation": {
            "design": "fixed stratified 80/20 hold-out; all preprocessing and tuning restricted to training data",
            "random_state": RANDOM_STATE,
            "train_n": int(len(x_train)),
            "test_n": int(len(x_test)),
            "test_positive_n": int(y_test.sum()),
            "test_prevalence": float(y_test.mean()),
            "metrics": metrics,
            "bootstrap_95_intervals": intervals,
        },
        "selected_terms_in_full_refit": coefficients,
        "input_reference": input_reference(analysis, features),
        "data_quality": {
            "hdl_label_disagreement": data_audit["hdl_label_disagreement"],
            "target_changes_if_hdl_recalculated": data_audit["target_changes_if_hdl_recalculated"],
            "warning": "Model follows the user-specified target column. Confirm that HDL correction was propagated to HDL异常 and 代谢异常 before clinical or publication use.",
        },
        "deployment_refit": "Hyperparameters and threshold frozen, then coefficients refit on all eligible participants.",
    }
    bundle = {
        "model": deploy_model,
        "threshold": threshold,
        "metadata": metadata,
    }
    output_path = output_dir / spec["filename"]
    joblib.dump(bundle, output_path, compress=3)

    return {
        "model_file": spec["filename"],
        "metadata": metadata,
        "tuning": tuning,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    audit = report["data_audit"]
    lines = [
        "# 横断面 LASSO Logistic 模型开发记录",
        "",
        "## 数据处理",
        "",
        f"- 源数据：{audit['source_rows']} 行，{audit['source_unique_participants']} 名唯一受试者。",
        f"- 源文件 SHA-256：`{audit['source_sha256']}`。",
        f"- 去重后分析样本：{audit['analysis_rows']} 名；阳性 {audit['outcome_positive']} 名（{audit['outcome_prevalence']:.1%}）。",
        f"- 重复体检号：{audit['duplicate_participant_ids']} 个；按 InBody BMI 与临床 BMI 差值最小的记录保留。",
        f"- 非生理腰臀比设为缺失并在训练管道内中位数插补：{audit['invalid_whr_values_set_missing']} 条。",
        f"- 结局核对：‘代谢异常’在 {audit['outcome_rule_matches']} 条分析记录中均等于血压异常、糖代谢异常、TG异常、HDL异常四项之和 ≥2。",
        f"- **重要质控提示：按当前 HDL-C 及男性 <1.0/女性 <1.3 mmol/L 重算时，HDL异常标签有 {audit['hdl_label_disagreement']} 条不一致，并会使代谢异常结局改变 {audit['target_changes_if_hdl_recalculated']} 条。当前模型仍严格遵照用户指定的‘代谢异常’列；临床使用或投稿前必须确认修正后的 HDL 是否已传播到标签。**",
        "- 上肢 ECW/TBW：（右上肢细胞外液+左上肢细胞外液）/（右上肢总水分+左上肢总水分）。",
        "- 运动频率：按网页既有分组重编码为 0（<1次/周）、1（1–2次/周）、2（3–5次/周）、3（>5次/周）。",
        f"- 源运动频率列含 {audit['raw_exercise_noninteger_values']} 个非整数值；当前按数值所在周频率区间分组，仍建议回溯其原始采集/插补过程。",
        f"- 年龄范围含 <18 岁 {audit['participants_younger_than_18']} 名、>80 岁 {audit['participants_older_than_80']} 名；若研究方案有成人或年龄上限，应按纳排标准重新训练。",
        "",
        "## 字段口径与映射",
        "",
        "为与原网页人口学基础模型保持一致，本次把‘人口学资料’暂按年龄、性别、运动、吸烟、饮酒五项解释；若您指严格人口学（仅年龄、性别），应重新训练。两个横断面模型统一使用临床 BMI 与腰围/臀围 WHR，保证模型 F 是模型 E 的嵌套简化版。",
        "",
        "| 用户/网页字段 | 本次建模来源或转换 |",
        "|---|---|",
    ]
    lines.extend(f"| {display} | {source} |" for display, source in REPORT_FIELD_MAPPING)
    lines.extend(
        [
            "",
        "## 建模与验证",
        "",
        "数值变量采用训练折中位数插补与标准化，分类变量采用众数插补、哑变量编码与标准化。",
        "L1 正则强度在训练集内以二项对数损失（binomial deviance）做分层 10 折交叉验证，并采用 1-SE 规则偏向更简约模型。",
        "分类阈值仅用训练集的 10 折折外预测按最大 Youden J 选取；最终性能来自固定的 20% 未触碰测试集。",
        "完成评估后，固定正则强度与阈值，用全部分析样本重拟合部署系数。",
        "",
        ]
    )
    for model in report["models"].values():
        meta = model["metadata"]
        metrics = meta["evaluation"]["metrics"]
        ci = meta["evaluation"]["bootstrap_95_intervals"]
        lines.extend(
            [
                f"## {meta['display_name']}",
                "",
                f"- 候选原始变量：{len(meta['candidate_features'])} 个；全样本重拟合后非零项：{len(meta['selected_terms_in_full_refit'])} 个。",
                f"- LASSO C：{meta['regularization']['selected_c']:.6g}；分类阈值：{meta['classification_threshold']:.3f}。",
                f"- 测试集 ROC AUC：{metrics['roc_auc']:.3f}（95% bootstrap CI {ci['roc_auc']['lower_95']:.3f}–{ci['roc_auc']['upper_95']:.3f}）。",
                f"- 测试集 PR AUC：{metrics['average_precision']:.3f}；Brier：{metrics['brier_score']:.3f}；Log loss：{metrics['log_loss']:.3f}。",
                f"- 校准截距：{metrics['calibration_intercept']:+.3f}；校准斜率：{metrics['calibration_slope']:.3f}。",
                f"- 阈值处灵敏度：{metrics['sensitivity']:.3f}；特异度：{metrics['specificity']:.3f}；PPV：{metrics['positive_predictive_value']:.3f}；NPV：{metrics['negative_predictive_value']:.3f}。",
                "- 非零项（标准化系数）：",
                "",
            ]
        )
        for row in meta["selected_terms_in_full_refit"]:
            lines.append(
                f"  - {row['transformed_feature']}: {row['standardized_coefficient']:+.4f}"
            )
        lines.append("")
    lines.extend(
        [
            "## 使用限制",
            "",
            "这些模型用于研究性横断面筛查，输出代表同一时点存在“代谢异常”的模型估计概率，不能解释为未来发病风险。",
            "目前性能属于单一数据集的内部留出验证；正式临床应用前仍需独立外部验证、校准检查和临床效用评估。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="横断面 Excel 工作簿路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="模型与报告输出目录（默认：脚本所在目录）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis, audit = prepare_analysis_data(args.source.resolve())
    models = {
        key: train_one_model(analysis, spec, output_dir, audit)
        for key, spec in MODEL_SPECS.items()
    }
    report = {
        "source_workbook": args.source.name,
        "sheet": "Sheet1",
        "development_date": date.today().isoformat(),
        "software_versions": software_versions(),
        "random_state": RANDOM_STATE,
        "source_column_map": SOURCE_COLUMN_MAP,
        "derived_source_columns": DERIVED_SOURCE_COLUMNS,
        "data_audit": audit,
        "models": models,
    }
    report_path = output_dir / "cross_sectional_model_report.json"
    report_path.write_text(
        json.dumps(to_builtin(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path = output_dir / "CROSS_SECTIONAL_MODEL_REPORT.md"
    markdown_path.write_text(render_markdown_report(to_builtin(report)), encoding="utf-8")
    print(f"Saved {len(models)} models and reports to {output_dir}")


if __name__ == "__main__":
    main()
