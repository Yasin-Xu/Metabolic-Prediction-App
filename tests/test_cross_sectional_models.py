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

        for model_name in model_names:
            with self.subTest(model=model_name):
                app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=30)
                app.sidebar.selectbox[0].select(model_name).run(timeout=30)
                app.button[0].click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)


if __name__ == "__main__":
    unittest.main()
