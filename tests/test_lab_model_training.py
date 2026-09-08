import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from train_lab_model_a import RAW_MISSINGNESS_COLUMNS, restore_original_missingness


class MissingnessRestorationTests(unittest.TestCase):
    def test_restores_only_original_missing_mask_without_changing_observed_values(self):
        current = pd.DataFrame(
            {
                "体检号": [101, 102],
                "糖化血红蛋白": [5.6, 5.7],
                "总胆固醇": [4.8, 5.1],
                "甘油三酯": [1.2, 1.4],
                "葡萄糖": [5.0, 5.3],
                # Integer dtype deliberately exercises the int-to-float promotion guard.
                "收缩压": [120, 130],
            }
        )
        original = current.copy()
        original.loc[1, ["糖化血红蛋白", "收缩压"]] = np.nan

        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_source = Path(temporary_directory) / "raw.xlsx"
            with pd.ExcelWriter(raw_source, engine="openpyxl") as writer:
                original.to_excel(writer, sheet_name="导出数据2", index=False)
            restored, audit = restore_original_missingness(current, raw_source)

        self.assertEqual(RAW_MISSINGNESS_COLUMNS[-1], "收缩压")
        self.assertTrue(pd.isna(restored.loc[1, "糖化血红蛋白"]))
        self.assertTrue(pd.isna(restored.loc[1, "收缩压"]))
        for field in RAW_MISSINGNESS_COLUMNS:
            self.assertEqual(restored.loc[0, field], current.loc[0, field])
        for field in ["总胆固醇", "甘油三酯", "葡萄糖"]:
            self.assertEqual(restored.loc[1, field], current.loc[1, field])
        self.assertEqual(audit["any_original_missing_rows"], 1)
        self.assertEqual(
            audit["fields"]["收缩压"]["current_values_restored_to_missing"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
