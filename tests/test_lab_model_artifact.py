import math
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_lab_model_a import MODEL_FEATURES, calculate_tyg


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_THRESHOLD = 0.2523321635423989


class LabModelArtifactTests(unittest.TestCase):
    def test_tyg_uses_mmol_l_to_mg_dl_conversion(self):
        triglyceride = 1.50
        fasting_glucose = 5.30
        expected = math.log((triglyceride * 88.5) * (fasting_glucose * 18) / 2)
        self.assertAlmostEqual(calculate_tyg(triglyceride, fasting_glucose), expected, places=12)

    def test_bundle_schema_and_default_prediction(self):
        bundle = joblib.load(PROJECT_DIR / "lab_model_a.pkl")
        self.assertEqual(set(bundle), {"model", "threshold", "metadata"})
        self.assertEqual(list(bundle["model"].feature_names_in_), MODEL_FEATURES)
        self.assertEqual(list(bundle["model"].classes_), [0, 1])
        self.assertGreater(bundle["threshold"], 0)
        self.assertLess(bundle["threshold"], 1)
        self.assertAlmostEqual(bundle["threshold"], EXPECTED_THRESHOLD, places=15)
        self.assertAlmostEqual(
            bundle["threshold"],
            bundle["metadata"]["classification_threshold"],
            places=15,
        )

        reference = bundle["metadata"]["input_reference"]
        row = {feature: reference[feature]["median"] for feature in MODEL_FEATURES}
        probabilities = bundle["model"].predict_proba(pd.DataFrame([row], columns=MODEL_FEATURES))[0]
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)

        quality = bundle["metadata"]["data_quality"]
        self.assertTrue(quality["outcome_rule_all_rows_match"])
        self.assertEqual(quality["outcome_rule_matches"], quality["analysis_rows"])
        self.assertGreater(quality["triglyceride_tyg_spearman"], 0.9)
        self.assertEqual(
            quality["feature_missing_counts"],
            {
                "体重指数": 0,
                "收缩压": 165,
                "下肢肌肉比率": 0,
                "细胞外液总量/身体总水分": 0,
                "上肢肌肉比率": 0,
                "下肢脂肪百分比": 0,
                "糖化血红蛋白": 18,
                "总胆固醇": 7,
                "甘油三酯": 7,
                "TyG 指数": 7,
            },
        )


if __name__ == "__main__":
    unittest.main()
