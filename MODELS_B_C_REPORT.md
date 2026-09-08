# Models B and C development record

## Data and split

- Source SHA-256: `86c9d287cfa6b74970ab03a4923576b58fcac984129f1b994242d7d1028c67bf`.
- Unique participants after deterministic deduplication: 3427.
- Positive outcome: 729 (21.3%).
- Fixed random state: 20260824; development/test split: 80%/20%.
- Smoking history and exercise frequency were excluded from both candidate sets before fitting.
- All preprocessing, feature screening, tuning, and threshold selection were confined to the development split.

## Selection

Model B retained the five prespecified basic fields and screened 12 body-composition candidates by repeated subsampling stability.
Body-composition retention threshold: 85%; selected: 上肢脂肪百分比, 下肢脂肪百分比, 身体总水分/去脂体重.
Model C retained the nonzero raw features from its development-set screen.

## 模型B（精简体成分）

- Inputs: 性别, 年龄, 饮酒史, 体重指数, 腰臀比, 上肢脂肪百分比, 下肢脂肪百分比, 身体总水分/去脂体重.
- Final C: 0.04075393.
- Frozen classification threshold: 0.25257970.
- Test ROC AUC: 0.798 (95% bootstrap CI 0.757–0.836).
- Test average precision: 0.536; Brier score: 0.136.
- Threshold sensitivity: 0.699; specificity: 0.748.
- Full-data refit standardized coefficients:

  - 腰臀比: +0.5883
  - 年龄: +0.3795
  - 下肢脂肪百分比: -0.3225
  - 上肢脂肪百分比: +0.1450
  - 身体总水分/去脂体重: -0.0973
  - 性别_1: +0.0555
  - 饮酒史_1: +0.0374
  - 体重指数: +0.0279

## 模型C（人口学基础）

- Inputs: 年龄, 饮酒史, 体重指数, 腰臀比.
- Final C: 0.033529241.
- Frozen classification threshold: 0.23437880.
- Test ROC AUC: 0.795 (95% bootstrap CI 0.753–0.832).
- Test average precision: 0.519; Brier score: 0.137.
- Threshold sensitivity: 0.760; specificity: 0.698.
- Full-data refit standardized coefficients:

  - 腰臀比: +0.6263
  - 年龄: +0.3712
  - 体重指数: +0.2923
  - 饮酒史_1: +0.0204

## Interpretation

The two artifacts estimate the probability of the recorded binary outcome. Performance is from an internal fixed hold-out and requires independent external validation before clinical deployment.
Model B's small ROC AUC difference versus Model C was not used to claim superiority; the three retained body-composition fields mainly improved average precision and Brier score in this split.
