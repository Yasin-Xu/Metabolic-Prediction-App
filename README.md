# 代谢异常在线预测系统

在线应用：<https://j35jfd5cfgnkmlf2ulytj9.streamlit.app/>

当前页面提供三个彼此独立的模型：

- **模型A（体成分+实验室）**：BMI、收缩压、体成分比例、糖化血红蛋白、总胆固醇、甘油三酯及系统计算的 TyG 指数。
- **模型E（人口学+体成分）**：人口学、生活方式、BMI、腰臀比和体成分指标。
- **模型F（人口学基础）**：人口学、生活方式、BMI 和腰臀比。

页面不再提供旧模型 B、C、D。三个模型分别输出代谢异常估计概率，并使用各自预先锁定的判别阈值；结果仅供研究性风险评估参考，不能替代临床诊断。

## 模型A输入与 TyG 计算

模型A固定使用以下工程特征：

1. 体重指数（kg/m²）
2. 收缩压（mmHg）
3. 下肢肌肉比率
4. 细胞外液总量/身体总水分（ECW/TBW）
5. 上肢肌肉比率
6. 下肢脂肪比率
7. 糖化血红蛋白（%）
8. 总胆固醇（mmol/L）
9. 甘油三酯（mmol/L）
10. TyG 指数

用户不需要手工填写 TyG。页面额外采集空腹血糖（mmol/L），并按下式自动计算：

```text
TyG = ln[(甘油三酯 mmol/L × 88.5) × (空腹血糖 mmol/L × 18) / 2]
```

体成分比例按 0–1 小数填写。下肢脂肪比率使用模型字段名 `下肢脂肪百分比`，实际含义为下肢脂肪量占全身体脂肪量的比例。

## 模型文件

- `lab_model_a.pkl`
- `cross_sectional_bodycomp_lasso.pkl`
- `cross_sectional_basic_lasso.pkl`

每个在线模型文件均保存预测管道、固定判别阈值和训练元数据。旧模型文件仍保留在 Git 历史及独立的 `original-version` 分支中，但不会出现在当前页面。

## 重新训练模型A

```bash
python train_lab_model_a.py "/path/to/data.xlsx" \
  --raw-source "/path/to/SS-导出初始数据集20240804.xlsx" \
  --output-dir .
```

`--raw-source` 仅用于按体检号恢复原始缺失掩码；脚本不会修改任一源工作簿，也不会用旧数据覆盖当前观测值。若该文件与主数据位于同一目录且文件名未变，可省略此参数。开发记录见 `LAB_MODEL_A_REPORT.md`，机器可读元数据见 `lab_model_a_report.json`。

## 测试

```bash
python -m unittest discover -s tests -v
```

建议使用与 `requirements.txt` 一致的依赖版本运行训练和部署。
