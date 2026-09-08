import hashlib
import math
import unittest
from pathlib import Path

import joblib
from streamlit.testing.v1 import AppTest


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_A_FEATURES = [
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

MODEL_B_FEATURES = [
    "性别",
    "年龄",
    "饮酒史",
    "体重指数",
    "腰臀比",
    "上肢脂肪百分比",
    "下肢脂肪百分比",
    "身体总水分/去脂体重",
]

MODEL_C_FEATURES = [
    "年龄",
    "饮酒史",
    "体重指数",
    "腰臀比",
]

EXPECTED_MODELS = {
    "模型A（体成分+实验室）": ("lab_model_a.pkl", MODEL_A_FEATURES),
    "模型B（精简体成分）": ("model_b.pkl", MODEL_B_FEATURES),
    "模型C（人口学基础）": ("model_c.pkl", MODEL_C_FEATURES),
}

FIXED_A_ARTIFACT = {
    "filename": "lab_model_a.pkl",
    "sha256": "9e65cf48730dcb36b6eb3b0fccd5c8f957520b6a0781ca3aa1d9a8402e65f5a5",
    "threshold": 0.2523321635423989,
}

EXPECTED_NEW_THRESHOLDS = {
    "model_b.pkl": 0.2525796984249065,
    "model_c.pkl": 0.2343787989709484,
}

FORBIDDEN_VISIBLE_TEXT = (
    "未来3年",
    "未来 3 年",
    "未来三年",
    "横断面",
    "随访",
    "筛查",
    "LASSO",
    "SVM",
    "XGBoost",
    "Logistic",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_text(app):
    pieces = []
    element_types = (
        "title",
        "header",
        "subheader",
        "caption",
        "markdown",
        "text",
        "info",
        "warning",
        "success",
        "error",
        "exception",
        "progress",
        "button",
        "selectbox",
        "number_input",
        "expander",
    )
    for element_type in element_types:
        for element in app.get(element_type):
            for attribute in ("label", "value", "text", "help", "message"):
                try:
                    value = getattr(element, attribute, None)
                except Exception:
                    continue
                if value is not None and not callable(value):
                    pieces.append(str(value))
            try:
                options = getattr(element, "options", None)
            except Exception:
                options = None
            if options is not None:
                pieces.extend(str(option) for option in options)
    return "\n".join(pieces)


class ArtifactTests(unittest.TestCase):
    def test_all_three_artifacts_have_the_expected_feature_order(self):
        for model_name, (filename, expected_features) in EXPECTED_MODELS.items():
            with self.subTest(model=model_name):
                bundle = joblib.load(PROJECT_DIR / filename)
                self.assertIn("model", bundle)
                self.assertIn("threshold", bundle)
                self.assertGreater(bundle["threshold"], 0)
                self.assertLess(bundle["threshold"], 1)
                self.assertEqual(
                    list(bundle["model"].feature_names_in_),
                    expected_features,
                )
                self.assertEqual(list(bundle["model"].classes_), [0, 1])

    def test_models_b_and_c_have_the_expected_thresholds(self):
        for filename, expected_threshold in EXPECTED_NEW_THRESHOLDS.items():
            with self.subTest(filename=filename):
                bundle = joblib.load(PROJECT_DIR / filename)
                self.assertAlmostEqual(
                    float(bundle["threshold"]),
                    expected_threshold,
                    places=14,
                )

    def test_model_a_artifact_and_threshold_are_unchanged(self):
        path = PROJECT_DIR / FIXED_A_ARTIFACT["filename"]
        self.assertEqual(file_sha256(path), FIXED_A_ARTIFACT["sha256"])
        bundle = joblib.load(path)
        self.assertAlmostEqual(
            float(bundle["threshold"]),
            FIXED_A_ARTIFACT["threshold"],
            places=14,
        )


class StreamlitTests(unittest.TestCase):
    def make_app(self):
        return AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=30)

    def assert_no_forbidden_visible_text(self, app):
        rendered = visible_text(app)
        for forbidden in FORBIDDEN_VISIBLE_TEXT:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden.casefold(), rendered.casefold())

    def test_selector_only_lists_models_a_b_and_c(self):
        app = self.make_app()
        self.assertEqual(
            list(app.sidebar.selectbox[0].options),
            list(EXPECTED_MODELS),
        )
        self.assertEqual(
            [item.label for item in app.expander],
            [
                "💡 结果如何理解",
                "📐 必读：所需指标及计算/获取方法",
                "ℹ️ 查看模型适用信息",
            ],
        )
        self.assert_no_forbidden_visible_text(app)

    def test_old_model_artifacts_are_not_registered(self):
        source = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("cross_sectional_bodycomp_lasso.pkl", source)
        self.assertNotIn("cross_sectional_basic_lasso.pkl", source)
        app = self.make_app()
        self.assertFalse(
            {"模型E（人口学+体成分）", "模型F（人口学基础）"}.intersection(
                app.sidebar.selectbox[0].options
            )
        )

    def test_all_registered_models_predict_from_default_form_values(self):
        for model_name in EXPECTED_MODELS:
            with self.subTest(model=model_name):
                app = self.make_app()
                app.sidebar.selectbox[0].select(model_name).run(timeout=30)
                self.assert_no_forbidden_visible_text(app)

                app.button[0].click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertFalse(
                    any(
                        "模型运行出错" in str(item.value)
                        or "模型暂时无法加载" in str(item.value)
                        for item in app.error
                    )
                )
                result_messages = [
                    str(item.value) for item in [*app.success, *app.error]
                ]
                self.assertTrue(
                    any(
                        "代谢异常风险较高" in message
                        or "代谢异常风险较低" in message
                        for message in result_messages
                    )
                )
                self.assertTrue(
                    any("模型固定判别阈值" in message for message in result_messages)
                )
                self.assertTrue(
                    any("不能替代临床诊断" in str(item.value) for item in app.info)
                )
                self.assert_no_forbidden_visible_text(app)

    def test_model_a_calculates_tyg_from_triglyceride_and_glucose(self):
        app = self.make_app()
        number_inputs = {item.label: item for item in app.number_input}
        self.assertFalse(any("TyG" in label for label in number_inputs))
        triglyceride_input = next(
            item for label, item in number_inputs.items() if label.startswith("甘油三酯")
        )
        glucose_input = next(
            item for label, item in number_inputs.items() if label.startswith("葡萄糖")
        )
        triglyceride_input.set_value(2.0)
        glucose_input.set_value(6.0)
        app.run(timeout=30)
        app.button[0].click().run(timeout=30)

        expected_tyg = math.log((2.0 * 88.5) * (6.0 * 18.0) / 2.0)
        self.assertAlmostEqual(
            app.session_state["last_computed_tyg"],
            expected_tyg,
            places=12,
        )
        self.assertEqual(len(app.exception), 0)

    def test_model_a_input_boundaries_are_enforced_and_accepted(self):
        expected_limits = {
            "体重指数": (10.0, 70.0),
            "收缩压": (80.0, 250.0),
            "下肢肌肉比率": (0.0001, 1.0),
            "上肢肌肉比率": (0.0001, 1.0),
            "ECW/TBW": (0.001, 1.0),
            "下肢脂肪比率": (0.0001, 1.0),
            "糖化血红蛋白": (3.0, 20.0),
            "甘油三酯": (0.01, 50.0),
            "总胆固醇": (0.01, 20.0),
            "葡萄糖": (0.01, 50.0),
        }

        for boundary_index in (0, 1):
            with self.subTest(boundary="minimum" if boundary_index == 0 else "maximum"):
                app = self.make_app()
                for label_prefix, limits in expected_limits.items():
                    item = next(
                        number_input
                        for number_input in app.number_input
                        if number_input.label.startswith(label_prefix)
                    )
                    self.assertEqual(item.min, limits[0])
                    self.assertEqual(item.max, limits[1])
                    item.set_value(limits[boundary_index])
                app.run(timeout=30)
                app.button[0].click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertFalse(
                    any("模型运行出错" in str(item.value) for item in app.error)
                )

    def test_models_b_and_c_use_the_configured_input_ranges(self):
        expected_limits = {
            "模型B（精简体成分）": {
                "年龄": (14.0, 100.0),
                "体重指数": (10.0, 70.0),
                "腰臀比": (0.5, 1.5),
                "上肢脂肪比率": (0.0, 0.4),
                "下肢脂肪比率": (0.05, 0.65),
                "身体总水分/去脂体重": (0.60, 0.85),
            },
            "模型C（人口学基础）": {
                "年龄": (14.0, 100.0),
                "体重指数": (10.0, 70.0),
                "腰臀比": (0.5, 1.5),
            },
        }
        for model_name, model_limits in expected_limits.items():
            with self.subTest(model=model_name):
                app = self.make_app()
                app.sidebar.selectbox[0].select(model_name).run(timeout=30)
                for label_prefix, limits in model_limits.items():
                    item = next(
                        number_input
                        for number_input in app.number_input
                        if number_input.label.startswith(label_prefix)
                    )
                    self.assertEqual((item.min, item.max), limits)
                app.run(timeout=30)
                app.button[0].click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertFalse(
                    any("模型运行出错" in str(item.value) for item in app.error)
                )

    def test_model_a_glucose_uses_artifact_reference_for_extrapolation_warning(self):
        bundle = joblib.load(PROJECT_DIR / "lab_model_a.pkl")
        auxiliary_reference = bundle["metadata"]["derived_input"][
            "auxiliary_input_reference"
        ]
        glucose_value = min(50.0, auxiliary_reference["percentile_99"] + 1.0)
        self.assertGreater(glucose_value, auxiliary_reference["percentile_99"])

        app = self.make_app()
        glucose_input = next(
            item for item in app.number_input if item.label.startswith("葡萄糖")
        )
        glucose_input.set_value(glucose_value)
        app.run(timeout=30)
        app.button[0].click().run(timeout=30)
        self.assertTrue(
            any(
                "葡萄糖（训练数据1%–99%" in str(item.value)
                for item in app.warning
            )
        )

    def test_models_b_and_c_show_their_fixed_thresholds(self):
        expected_threshold_text = {
            "模型B（精简体成分）": "模型固定判别阈值 25.3%",
            "模型C（人口学基础）": "模型固定判别阈值 23.4%",
        }
        for model_name, threshold_text in expected_threshold_text.items():
            with self.subTest(model=model_name):
                app = self.make_app()
                app.sidebar.selectbox[0].select(model_name).run(timeout=30)
                app.button[0].click().run(timeout=30)
                result_messages = [
                    str(item.value) for item in [*app.success, *app.error]
                ]
                self.assertTrue(
                    any(threshold_text in message for message in result_messages)
                )

    def test_default_probabilities_match_the_frozen_artifacts(self):
        expected_result_text = {
            "模型A（体成分+实验室）": "模型估计概率 10.78% < 模型固定判别阈值 25.2%",
            "模型B（精简体成分）": "模型估计概率 20.70% < 模型固定判别阈值 25.3%",
            "模型C（人口学基础）": "模型估计概率 19.95% < 模型固定判别阈值 23.4%",
        }
        for model_name, expected_text in expected_result_text.items():
            with self.subTest(model=model_name):
                app = self.make_app()
                app.sidebar.selectbox[0].select(model_name).run(timeout=30)
                app.button[0].click().run(timeout=30)
                result_messages = [
                    str(item.value) for item in [*app.success, *app.error]
                ]
                self.assertTrue(
                    any(expected_text in message for message in result_messages)
                )


if __name__ == "__main__":
    unittest.main()
