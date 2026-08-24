import sys
import unittest
from pathlib import Path

import joblib
from streamlit.testing.v1 import AppTest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from train_cross_sectional_models import MODEL_SPECS  # noqa: E402


class CrossSectionalArtifactTests(unittest.TestCase):
    def test_artifacts_match_training_specs(self):
        for spec in MODEL_SPECS.values():
            with self.subTest(model=spec["display_name"]):
                bundle = joblib.load(PROJECT_DIR / spec["filename"])
                self.assertIn("model", bundle)
                self.assertIn("threshold", bundle)
                self.assertGreater(bundle["threshold"], 0)
                self.assertLess(bundle["threshold"], 1)
                self.assertEqual(
                    list(bundle["model"].feature_names_in_),
                    list(spec["features"]),
                )
                self.assertEqual(list(bundle["model"].classes_), [0, 1])


class StreamlitSmokeTests(unittest.TestCase):
    def test_all_registered_models_predict_from_default_form_values(self):
        app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=30)
        model_names = list(app.sidebar.selectbox[0].options)
        self.assertEqual(len(model_names), 6)
        for method_name in ("LASSO", "SVM", "XGBoost", "Logistic"):
            self.assertFalse(any(method_name in name for name in model_names))

        for model_name in model_names:
            with self.subTest(model=model_name):
                app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=30)
                app.sidebar.selectbox[0].select(model_name).run(timeout=30)
                app.button[0].click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)

    def test_cross_sectional_quality_and_validation_details_are_hidden(self):
        app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=30)
        app.sidebar.selectbox[0].select(
            "横断面模型E（人口学+体成分）"
        ).run(timeout=30)

        expander_labels = [item.label for item in app.expander]
        self.assertNotIn("🧭 横断面模型说明与内部验证表现", expander_labels)
        self.assertFalse(any("质控" in str(item.value) for item in app.warning))
        self.assertEqual(len(app.metric), 0)
        rendered_markdown = "\n".join(str(item.value) for item in app.markdown)
        self.assertIn("总胆固醇异常", rendered_markdown)
        self.assertNotIn("甘油三酯异常", rendered_markdown)

        app.button[0].click().run(timeout=30)
        expander_labels = [item.label for item in app.expander]
        self.assertNotIn("查看该横断面模型的内部验证表现", expander_labels)
        self.assertEqual(len(app.metric), 0)

    def test_basic_model_uses_one_fixed_threshold_for_both_result_branches(self):
        model_name = "横断面模型F（人口学基础）"

        negative_app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=30)
        negative_app.sidebar.selectbox[0].select(model_name).run(timeout=30)
        negative_app.button[0].click().run(timeout=30)
        self.assertIn("< 模型固定判别阈值 23.3%", negative_app.success[0].value)

        positive_app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=30)
        positive_app.sidebar.selectbox[0].select(model_name).run(timeout=30)
        positive_app.number_input[0].set_value(80)
        positive_app.number_input[1].set_value(40)
        positive_app.number_input[2].set_value(1.20)
        positive_app.run(timeout=30)
        positive_app.button[0].click().run(timeout=30)
        self.assertIn("≥ 模型固定判别阈值 23.3%", positive_app.error[0].value)

        bodycomp_app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=30)
        bodycomp_app.sidebar.selectbox[0].select(
            "横断面模型E（人口学+体成分）"
        ).run(timeout=30)
        bodycomp_app.button[0].click().run(timeout=30)
        self.assertIn("< 模型固定判别阈值 25.3%", bodycomp_app.success[0].value)


if __name__ == "__main__":
    unittest.main()
