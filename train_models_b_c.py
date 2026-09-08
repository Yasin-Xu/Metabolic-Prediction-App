"""Retrain the streamlined Model B and Model C deployment artifacts.

The source workbook is read-only. Model B keeps the prespecified demographic,
anthropometric core after removing smoking and exercise, then retains only body-
composition variables that are selected in at least 85% of repeated development-
set subsamples. Model C screens the reduced basic candidate set with L1 logistic
regression. Hyperparameter tuning, selection, and threshold estimation are all
restricted to the development split before evaluation on the untouched test set.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
    train_test_split,
)

from train_cross_sectional_models import (
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TARGET,
    bootstrap_intervals,
    calibration_parameters,
    choose_threshold,
    coefficient_table,
    input_reference,
    make_pipeline,
    metric_values,
    prepare_analysis_data,
    select_regularization_strength,
    software_versions,
    to_builtin,
)


BASIC_CANDIDATES = ["性别", "年龄", "饮酒史", "体重指数", "腰臀比"]

BODY_COMPOSITION_CANDIDATES = [
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

MODEL_B_CANDIDATES = BASIC_CANDIDATES + BODY_COMPOSITION_CANDIDATES
MODEL_C_CANDIDATES = BASIC_CANDIDATES

EXPECTED_MODEL_B_BODY_FEATURES = [
    "上肢脂肪百分比",
    "下肢脂肪百分比",
    "身体总水分/去脂体重",
]
EXPECTED_MODEL_C_FEATURES = ["年龄", "饮酒史", "体重指数", "腰臀比"]

STABILITY_ITERATIONS = 300
STABILITY_SUBSAMPLE_FRACTION = 0.80
STABILITY_MINIMUM = 0.85

MODEL_SPECS = {
    "model_b": {
        "display_name": "模型B（精简体成分）",
        "filename": "model_b.pkl",
    },
    "model_c": {
        "display_name": "模型C（人口学基础）",
        "filename": "model_c.pkl",
    },
}


def transformed_term_to_raw(term: str, raw_features: list[str]) -> str:
    """Map a pipeline output name back to its source feature."""
    for feature in sorted(raw_features, key=len, reverse=True):
        if term == feature or term.startswith(f"{feature}_"):
            return feature
    raise ValueError(f"无法把转换后字段映射回原始字段: {term}")


def selected_raw_features(model, raw_features: list[str]) -> list[str]:
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    selected = {
        transformed_term_to_raw(str(name), raw_features)
        for name, coefficient in zip(names, coefficients, strict=True)
        if not np.isclose(coefficient, 0.0, atol=1e-10)
    }
    return [feature for feature in raw_features if feature in selected]


def stability_selection(
    x_train,
    y_train,
    features: list[str],
    selected_c: float,
) -> dict[str, float]:
    """Estimate raw-feature selection frequency in repeated 80% subsamples."""
    splitter = StratifiedShuffleSplit(
        n_splits=STABILITY_ITERATIONS,
        train_size=STABILITY_SUBSAMPLE_FRACTION,
        random_state=RANDOM_STATE,
    )
    counts = {feature: 0 for feature in features}
    for subsample_indices, _ in splitter.split(x_train, y_train):
        model = make_pipeline(features, selected_c).fit(
            x_train.iloc[subsample_indices],
            y_train.iloc[subsample_indices],
        )
        for feature in selected_raw_features(model, features):
            counts[feature] += 1
    return {
        feature: count / STABILITY_ITERATIONS
        for feature, count in counts.items()
    }


def evaluate_and_refit(
    analysis,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    features: list[str],
    display_name: str,
    filename: str,
    output_dir: Path,
    data_audit: dict[str, Any],
    selection_metadata: dict[str, Any],
) -> dict[str, Any]:
    x_train = analysis.iloc[train_indices][features]
    y_train = analysis.iloc[train_indices][TARGET]
    x_test = analysis.iloc[test_indices][features]
    y_test = analysis.iloc[test_indices][TARGET]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    selected_c, tuning = select_regularization_strength(x_train, y_train, features, cv)
    pipeline = make_pipeline(features, selected_c)
    out_of_fold_probabilities = cross_val_predict(
        clone(pipeline),
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold = choose_threshold(y_train, out_of_fold_probabilities)

    evaluation_model = clone(pipeline).fit(x_train, y_train)
    test_probabilities = evaluation_model.predict_proba(x_test)[:, 1]
    metrics = metric_values(y_test, test_probabilities, threshold)
    metrics.update(calibration_parameters(y_test, test_probabilities))
    intervals = bootstrap_intervals(y_test, test_probabilities, threshold)

    deploy_model = clone(pipeline).fit(analysis[features], analysis[TARGET])
    nonzero_terms = coefficient_table(deploy_model)
    deployed_raw_features = selected_raw_features(deploy_model, features)
    if deployed_raw_features != features:
        raise RuntimeError(
            f"{display_name}全样本重拟合后存在零系数字段: "
            f"{sorted(set(features) - set(deployed_raw_features))}"
        )

    metadata = {
        "display_name": display_name,
        "development_date": date.today().isoformat(),
        "software_versions": software_versions(),
        "source_sha256": data_audit["source_sha256"],
        "outcome": TARGET,
        "positive_class": 1,
        "features": features,
        "categorical_features": [
            feature for feature in features if feature in CATEGORICAL_FEATURES
        ],
        "selection": selection_metadata,
        "regularization": {
            "penalty": "L1",
            "solver": "liblinear",
            "selected_c": selected_c,
        },
        "classification_threshold": threshold,
        "threshold_method": "maximum Youden J on development-set 10-fold out-of-fold predictions",
        "evaluation": {
            "design": "fixed stratified 80/20 hold-out; preprocessing, screening, tuning, and threshold selection restricted to development data",
            "random_state": RANDOM_STATE,
            "train_n": int(len(train_indices)),
            "test_n": int(len(test_indices)),
            "test_positive_n": int(y_test.sum()),
            "test_prevalence": float(y_test.mean()),
            "metrics": metrics,
            "bootstrap_95_intervals": intervals,
        },
        "selected_terms_in_full_refit": nonzero_terms,
        "input_reference": input_reference(analysis, features),
        "deployment_refit": "Hyperparameter and threshold frozen before refitting coefficients on all eligible participants.",
    }
    bundle = {"model": deploy_model, "threshold": threshold, "metadata": metadata}
    joblib.dump(bundle, output_dir / filename, compress=3)
    return {
        "model_file": filename,
        "metadata": metadata,
        "tuning": tuning,
    }


def train_models(analysis, data_audit: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    all_indices = np.arange(len(analysis))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.20,
        stratify=analysis[TARGET],
        random_state=RANDOM_STATE,
    )
    development = analysis.iloc[train_indices].reset_index(drop=True)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    model_b_screen_c, model_b_screen_tuning = select_regularization_strength(
        development[MODEL_B_CANDIDATES],
        development[TARGET],
        MODEL_B_CANDIDATES,
        cv,
    )
    model_b_frequencies = stability_selection(
        development[MODEL_B_CANDIDATES],
        development[TARGET],
        MODEL_B_CANDIDATES,
        model_b_screen_c,
    )
    selected_body_features = [
        feature
        for feature in BODY_COMPOSITION_CANDIDATES
        if model_b_frequencies[feature] >= STABILITY_MINIMUM
    ]
    if selected_body_features != EXPECTED_MODEL_B_BODY_FEATURES:
        raise RuntimeError(
            "模型B体成分稳定筛选结果与预期不一致: "
            f"实际={selected_body_features}, 预期={EXPECTED_MODEL_B_BODY_FEATURES}"
        )
    model_b_features = BASIC_CANDIDATES + selected_body_features

    model_c_screen_c, model_c_screen_tuning = select_regularization_strength(
        development[MODEL_C_CANDIDATES],
        development[TARGET],
        MODEL_C_CANDIDATES,
        cv,
    )
    model_c_screen = make_pipeline(MODEL_C_CANDIDATES, model_c_screen_c).fit(
        development[MODEL_C_CANDIDATES],
        development[TARGET],
    )
    model_c_features = selected_raw_features(model_c_screen, MODEL_C_CANDIDATES)
    if model_c_features != EXPECTED_MODEL_C_FEATURES:
        raise RuntimeError(
            "模型C筛选结果与预期不一致: "
            f"实际={model_c_features}, 预期={EXPECTED_MODEL_C_FEATURES}"
        )

    model_b_selection = {
        "candidate_features": MODEL_B_CANDIDATES,
        "prespecified_core_features": BASIC_CANDIDATES,
        "body_composition_candidates": BODY_COMPOSITION_CANDIDATES,
        "body_composition_selection_rule": (
            f"selected in at least {STABILITY_MINIMUM:.0%} of "
            f"{STABILITY_ITERATIONS} repeated stratified "
            f"{STABILITY_SUBSAMPLE_FRACTION:.0%} development-set subsamples"
        ),
        "screening_c": model_b_screen_c,
        "selection_frequency": model_b_frequencies,
        "selected_body_composition_features": selected_body_features,
        "screening_tuning": model_b_screen_tuning,
    }
    model_c_selection = {
        "candidate_features": MODEL_C_CANDIDATES,
        "selection_rule": "nonzero raw features in the development-set one-standard-error L1 fit",
        "screening_c": model_c_screen_c,
        "selected_features": model_c_features,
        "screening_tuning": model_c_screen_tuning,
    }

    return {
        "model_b": evaluate_and_refit(
            analysis,
            train_indices,
            test_indices,
            model_b_features,
            MODEL_SPECS["model_b"]["display_name"],
            MODEL_SPECS["model_b"]["filename"],
            output_dir,
            data_audit,
            model_b_selection,
        ),
        "model_c": evaluate_and_refit(
            analysis,
            train_indices,
            test_indices,
            model_c_features,
            MODEL_SPECS["model_c"]["display_name"],
            MODEL_SPECS["model_c"]["filename"],
            output_dir,
            data_audit,
            model_c_selection,
        ),
    }


def render_report(report: dict[str, Any]) -> str:
    audit = report["data_audit"]
    lines = [
        "# Models B and C development record",
        "",
        "## Data and split",
        "",
        f"- Source SHA-256: `{audit['source_sha256']}`.",
        f"- Unique participants after deterministic deduplication: {audit['analysis_rows']}.",
        f"- Positive outcome: {audit['outcome_positive']} ({audit['outcome_prevalence']:.1%}).",
        f"- Fixed random state: {RANDOM_STATE}; development/test split: 80%/20%.",
        "- Smoking history and exercise frequency were excluded from both candidate sets before fitting.",
        "- All preprocessing, feature screening, tuning, and threshold selection were confined to the development split.",
        "",
        "## Selection",
        "",
        "Model B retained the five prespecified basic fields and screened 12 body-composition candidates by repeated subsampling stability.",
        f"Body-composition retention threshold: {STABILITY_MINIMUM:.0%}; selected: "
        + ", ".join(report["models"]["model_b"]["metadata"]["selection"]["selected_body_composition_features"])
        + ".",
        "Model C retained the nonzero raw features from its development-set screen.",
        "",
    ]
    for key in ("model_b", "model_c"):
        metadata = report["models"][key]["metadata"]
        metrics = metadata["evaluation"]["metrics"]
        intervals = metadata["evaluation"]["bootstrap_95_intervals"]
        lines.extend(
            [
                f"## {metadata['display_name']}",
                "",
                "- Inputs: " + ", ".join(metadata["features"]) + ".",
                f"- Final C: {metadata['regularization']['selected_c']:.8g}.",
                f"- Frozen classification threshold: {metadata['classification_threshold']:.8f}.",
                f"- Test ROC AUC: {metrics['roc_auc']:.3f} "
                f"(95% bootstrap CI {intervals['roc_auc']['lower_95']:.3f}–{intervals['roc_auc']['upper_95']:.3f}).",
                f"- Test average precision: {metrics['average_precision']:.3f}; Brier score: {metrics['brier_score']:.3f}.",
                f"- Threshold sensitivity: {metrics['sensitivity']:.3f}; specificity: {metrics['specificity']:.3f}.",
                "- Full-data refit standardized coefficients:",
                "",
            ]
        )
        lines.extend(
            f"  - {row['transformed_feature']}: {row['standardized_coefficient']:+.4f}"
            for row in metadata["selected_terms_in_full_refit"]
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The two artifacts estimate the probability of the recorded binary outcome. Performance is from an internal fixed hold-out and requires independent external validation before clinical deployment.",
            "Model B's small ROC AUC difference versus Model C was not used to claim superiority; the three retained body-composition fields mainly improved average precision and Brier score in this split.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Source Excel workbook")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Artifact/report directory (default: script directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis, data_audit = prepare_analysis_data(args.source.resolve())
    models = train_models(analysis, data_audit, output_dir)
    report = {
        "source_workbook": args.source.name,
        "development_date": date.today().isoformat(),
        "software_versions": software_versions(),
        "random_state": RANDOM_STATE,
        "data_audit": data_audit,
        "models": models,
    }
    serializable_report = to_builtin(report)
    (output_dir / "models_b_c_report.json").write_text(
        json.dumps(serializable_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "MODELS_B_C_REPORT.md").write_text(
        render_report(serializable_report),
        encoding="utf-8",
    )
    print(f"Saved Model B, Model C, and reports to {output_dir}")


if __name__ == "__main__":
    main()
