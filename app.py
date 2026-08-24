from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="代谢异常预测系统", page_icon="🩺", layout="centered")


# 模型中的字段名均使用训练时保存的原始名称，展示名称单独放在 FEATURE_DICT 中。
MODEL_REGISTRY = {
    "随访模型A（全变量 LASSO）": {
        "file": "lasso_model.pkl",
        "endpoint": "follow_up",
        "features": [
            "下肢肌肉比率",
            "TyG 指数",
            "糖化血红蛋白",
            "体重指数",
            "总胆固醇",
            "尿素",
            "细胞外液总量/身体总水分",
            "γ-谷氨酰转移酶",
            "甘油三酯",
            "白蛋白",
            "嗜酸性粒细胞百分比",
            "收缩压",
            "嗜碱性粒细胞百分比",
            "血红蛋白",
            "上肢肌肉比率",
            "下肢脂肪百分比",
        ],
        "description": "全量信息条件下的 3 年随访风险模型。",
    },
    "随访模型B（体成分 SVM）": {
        "file": "svm_model.pkl",
        "endpoint": "follow_up",
        "features": [
            "年龄",
            "体重指数",
            "腰臀比",
            "体脂肪",
            "体脂百分比",
            "上肢肌肉比率",
            "躯干肌肉量比率",
            "下肢肌肉比率",
            "躯干脂肪百分比",
            "下肢脂肪百分比",
            "细胞外液总量/身体总水分",
            "吸烟史",
            "身体总水分",
            "身体总水分/去脂体重",
        ],
        "description": "无需抽血、依赖体成分数据的 3 年随访风险模型。",
    },
    "随访模型C（临床常规 XGBoost）": {
        "file": "xgb_model.pkl",
        "endpoint": "follow_up",
        "features": [
            "甘油三酯",
            "糖化血红蛋白",
            "体重指数",
            "TyG 指数",
            "尿酸",
            "门冬氨酸氨基转移酶",
            "年龄",
            "尿素",
            "嗜酸性粒细胞百分比",
            "肌酐",
            "血红蛋白",
            "总胆固醇",
            "嗜碱性粒细胞百分比",
            "丙氨酸氨基转移酶",
            "脂蛋白α测定",
            "红细胞计数",
            "舒张压",
            "白细胞计数",
            "收缩压",
            "γ-谷氨酰转移酶",
            "白蛋白",
            "腰臀比",
        ],
        "description": "使用常规体检和生化数据的 3 年随访风险模型。",
    },
    "随访模型D（人口学基础 Logistic）": {
        "file": "lr_model.pkl",
        "endpoint": "follow_up",
        "features": ["性别", "年龄", "运动频率", "吸烟史", "饮酒史", "体重指数", "腰臀比"],
        "description": "仅使用人口学、生活方式、BMI 和腰臀比的 3 年随访基准模型。",
    },
    "横断面模型E（人口学+体成分 LASSO）": {
        "file": "cross_sectional_bodycomp_lasso.pkl",
        "endpoint": "cross_sectional",
        "features": [
            "性别",
            "年龄",
            "运动频率",
            "吸烟史",
            "饮酒史",
            "体重指数",
            "腰臀比",
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
        ],
        "description": "使用人口学、生活方式、BMI、腰臀比和无创体成分指标，判断当前代谢异常。",
        "use_case": "适合已经完成生物电阻抗/体成分检测的受试者，用于考察体成分信息是否可能提供额外判别信息。",
        "measurement_requirement": "需要年龄、性别和生活方式信息，身高/体重、腰围/臀围，以及同一次体成分检测报告。",
    },
    "横断面模型F（人口学基础 LASSO）": {
        "file": "cross_sectional_basic_lasso.pkl",
        "endpoint": "cross_sectional",
        "features": ["性别", "年龄", "运动频率", "吸烟史", "饮酒史", "体重指数", "腰臀比"],
        "description": "仅使用人口学、生活方式、BMI 和腰臀比，判断当前代谢异常。",
        "use_case": "适合没有体成分仪时进行简化筛查，也可作为横断面模型 E 的基础对照模型。",
        "measurement_requirement": "仅需年龄、性别和生活方式信息，以及身高、体重、腰围和臀围。",
    },
}


OPTIONS_MAP = {
    "性别": {"男性": 1, "女性": 0},
    "吸烟史": {"无吸烟史": 0, "有吸烟史（含已戒烟）": 1},
    "饮酒史": {"否认饮酒": 0, "有饮酒史": 1},
    "运动频率": {
        "极少（<1次/周）": 0,
        "偶尔（1–2次/周）": 1,
        "规律（3–5次/周）": 2,
        "经常（>5次/周）": 3,
    },
}

CATEGORY_ORDER = [
    "👤 基本人口学及生活方式",
    "📏 体格检查指标",
    "⚖️ 体成分指标",
    "🩸 实验室指标",
]


def feature(
    category,
    unit="",
    default=0.0,
    *,
    label=None,
    decimals=2,
    model_scale=1.0,
    max_value=None,
):
    return {
        "cat": category,
        "unit": unit,
        "def": default,
        "label": label,
        "decimals": decimals,
        "model_scale": model_scale,
        "max": max_value,
    }


DEMOGRAPHIC = "👤 基本人口学及生活方式"
PHYSICAL = "📏 体格检查指标"
BODY = "⚖️ 体成分指标"
LAB = "🩸 实验室指标"

FEATURE_DICT = {
    "年龄": feature(DEMOGRAPHIC, "岁", 53.0, decimals=0, max_value=120.0),
    "性别": feature(DEMOGRAPHIC),
    "运动频率": feature(DEMOGRAPHIC),
    "吸烟史": feature(DEMOGRAPHIC),
    "饮酒史": feature(DEMOGRAPHIC),
    "体重指数": feature(PHYSICAL, "kg/m²", 25.0, max_value=100.0),
    "腰臀比": feature(PHYSICAL, "WHR", 0.93, decimals=2, max_value=2.0),
    "收缩压": feature(PHYSICAL, "mmHg", 120.0, decimals=0),
    "舒张压": feature(PHYSICAL, "mmHg", 80.0, decimals=0),
    "体脂肪": feature(BODY, "kg", 19.1, label="体脂肪量"),
    "体脂百分比": feature(BODY, "%", 28.3, max_value=100.0),
    "上肢肌肉比率": feature(BODY, "上肢肌肉量/全身肌肉量", 0.116, decimals=4, max_value=1.0),
    "躯干肌肉量比率": feature(BODY, "躯干肌肉量/全身肌肉量", 0.484, label="躯干肌肉比率", decimals=4, max_value=1.0),
    "下肢肌肉比率": feature(BODY, "下肢肌肉量/全身肌肉量", 0.317, decimals=4, max_value=1.0),
    "细胞外液总量/身体总水分": feature(BODY, "比值", 0.378, label="ECW/TBW", decimals=3, max_value=1.0),
    "上肢细胞外液总量/身体总水分": feature(
        BODY,
        "左右上肢合并比值",
        0.378,
        label="上肢 ECW/TBW",
        decimals=4,
        max_value=1.0,
    ),
    "上肢脂肪百分比": feature(BODY, "上肢脂肪量/全身体脂肪量", 0.131, label="上肢脂肪比率", decimals=4, max_value=1.0),
    "躯干脂肪百分比": feature(BODY, "躯干脂肪量/全身体脂肪量", 0.527, label="躯干脂肪比率", decimals=4, max_value=1.0),
    "下肢脂肪百分比": feature(BODY, "下肢脂肪量/全身体脂肪量", 0.282, label="下肢脂肪比率", decimals=4, max_value=1.0),
    "身体总水分": feature(BODY, "L（按体成分报告）", 36.7),
    "身体总水分/去脂体重": feature(
        BODY,
        "输入0–1比值，如0.734",
        0.734,
        label="身体总水分/去脂体重（TBW/FFM）",
        decimals=3,
        model_scale=100.0,
        max_value=1.0,
    ),
    "甘油三酯": feature(LAB, "mmol/L", 1.5),
    "糖化血红蛋白": feature(LAB, "%", 5.5),
    "TyG 指数": feature(LAB, "", 8.5),
    "尿酸": feature(LAB, "μmol/L", 300.0),
    "门冬氨酸氨基转移酶": feature(LAB, "U/L", 20.0),
    "尿素": feature(LAB, "mmol/L", 5.0),
    "嗜酸性粒细胞百分比": feature(LAB, "%", 2.0),
    "肌酐": feature(LAB, "μmol/L", 70.0),
    "血红蛋白": feature(LAB, "g/L", 135.0),
    "总胆固醇": feature(LAB, "mmol/L", 4.5),
    "嗜碱性粒细胞百分比": feature(LAB, "%", 0.5),
    "丙氨酸氨基转移酶": feature(LAB, "U/L", 20.0),
    "脂蛋白α测定": feature(LAB, "mg/L", 150.0),
    "红细胞计数": feature(LAB, "10^12/L", 4.5),
    "白细胞计数": feature(LAB, "10^9/L", 6.0),
    "γ-谷氨酰转移酶": feature(LAB, "U/L", 25.0),
    "白蛋白": feature(LAB, "g/L", 45.0),
}


# 横断面模型输入说明。所有体成分指标应来自同一次检测，避免混用不同日期的报告。
FEATURE_GUIDANCE = {
    "性别": "按本研究编码选择：男性=1，女性=0。",
    "年龄": "以检查日期减出生日期计算周岁，填写完整岁数。",
    "运动频率": "按平均每周运动次数分组：<1次、1–2次、3–5次或>5次。",
    "吸烟史": "从未吸烟选择“无”；当前吸烟或既往吸烟/已戒烟均选择“有”。",
    "饮酒史": "按研究问卷填写：否认饮酒=0；任何有饮酒记录（包括应酬性饮酒）=1。",
    "体重指数": "BMI = 体重(kg) ÷ 身高(m)²。例如70 kg、1.75 m，BMI=22.86 kg/m²。模型训练使用临床身高、体重计算的“体重指数”，不是InBody的“身体质量指数”。",
    "腰臀比": "WHR = 腰围 ÷ 臀围。两者使用相同单位，按原表训练口径保留2位小数；例如腰围85 cm、臀围95 cm，填写0.89。模型训练使用临床腰围、臀围计算的“腰臀比”，不是“腰臀比INBODY”。",
    "身体总水分": "从同一次生物电阻抗/体成分检测报告读取身体总水分（TBW），按报告数值填写，通常以L表示。",
    "体脂肪": "从同一次体成分报告读取体脂肪量，单位kg。",
    "体脂百分比": "优先读取同一次体成分报告的体脂百分比；无报告值时可用体脂肪量 ÷ 体重 ×100%近似计算。填写百分数点，例如28.3%填写28.3。",
    "上肢肌肉比率": "(右上肢肌肉量 + 左上肢肌肉量) ÷ 全身肌肉量；分母是报告中的“肌肉量”，不是“骨骼肌量”。填写0–1小数，例如0.116。",
    "躯干肌肉量比率": "躯干肌肉量 ÷ 全身肌肉量；分母是报告中的“肌肉量”，不是“骨骼肌量”。填写0–1小数，例如0.484。",
    "下肢肌肉比率": "(右下肢肌肉量 + 左下肢肌肉量) ÷ 全身肌肉量；分母是报告中的“肌肉量”，不是“骨骼肌量”。填写0–1小数，例如0.317。",
    "细胞外液总量/身体总水分": "全身ECW/TBW = 全身细胞外液总量 ÷ 身体总水分；填写0–1小数，例如0.378。",
    "上肢细胞外液总量/身体总水分": "上肢ECW/TBW = (右上肢ECW + 左上肢ECW) ÷ (右上肢TBW + 左上肢TBW)，不是左右比值的简单平均。",
    "上肢脂肪百分比": "(右上肢脂肪量 + 左上肢脂肪量) ÷ 全身体脂肪量；填写0–1小数，例如0.131，而不是13.1。",
    "躯干脂肪百分比": "躯干脂肪量 ÷ 全身体脂肪量；填写0–1小数，例如0.527，而不是52.7。",
    "下肢脂肪百分比": "(右下肢脂肪量 + 左下肢脂肪量) ÷ 全身体脂肪量；填写0–1小数，例如0.282，而不是28.2。",
    "身体总水分/去脂体重": "TBW/FFM = 身体总水分 ÷ 去脂体重。网页填写0–1小数，例如报告为73.4%时填写0.734。",
}

TERM_LABELS = {
    "性别_1.0": "性别（男性 vs 女性）",
    "吸烟史_1.0": "有吸烟史（vs 无）",
    "饮酒史_1.0": "有饮酒史（vs 无）",
    "运动频率_1.0": "运动频率：1–2次/周（vs <1次/周）",
    "运动频率_2.0": "运动频率：3–5次/周（vs <1次/周）",
    "运动频率_3.0": "运动频率：>5次/周（vs <1次/周）",
}


@st.cache_resource(show_spinner=False)
def load_model_bundle(filename):
    artifact = joblib.load(APP_DIR / filename)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {"model": artifact, "threshold": None, "metadata": {}}


def display_label(feature_name):
    info = FEATURE_DICT[feature_name]
    label = info["label"] or feature_name
    return f"{label}（{info['unit']}）" if info["unit"] else label


def display_name(feature_name):
    info = FEATURE_DICT[feature_name]
    return info["label"] or feature_name


def selected_term_label(term_name):
    if term_name in TERM_LABELS:
        return TERM_LABELS[term_name]
    if term_name in FEATURE_DICT:
        return display_name(term_name)
    return term_name


def show_cross_sectional_model_overview(config):
    with st.expander("🧭 横断面模型说明与内部验证表现", expanded=True):
        st.markdown(
            f"**用途**：{config['use_case']}  \n"
            f"**测量要求**：{config['measurement_requirement']}  \n"
            "**结局含义**：输出为当前检查时点存在代谢异常的估计概率，不是未来发病风险。"
        )

        model_path = APP_DIR / config["file"]
        if not model_path.exists():
            st.warning(f"尚未找到模型文件：{model_path.name}")
            return

        bundle = load_model_bundle(config["file"])
        metadata = bundle.get("metadata", {})
        evaluation = metadata.get("evaluation", {})
        metrics = evaluation.get("metrics", {})
        if metrics:
            columns = st.columns(4)
            columns[0].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
            columns[1].metric("PR AUC", f"{metrics['average_precision']:.3f}")
            columns[2].metric("灵敏度", f"{metrics['sensitivity']:.1%}")
            columns[3].metric("特异度", f"{metrics['specificity']:.1%}")
            st.caption(
                f"固定20%内部测试集（n={evaluation.get('test_n', '—')}）；"
                f"模型判别阈值为 {float(bundle.get('threshold') or 0.5):.1%}。"
                "这些指标尚未经过独立外部验证。"
            )

        selected_terms = metadata.get("selected_terms_in_full_refit", [])
        if selected_terms:
            labels = [selected_term_label(row["transformed_feature"]) for row in selected_terms]
            st.markdown("**LASSO 全样本重拟合后的非零项**：" + "、".join(labels) + "。")
            st.caption("非零系数表示进入最终预测方程，不代表因果关系。页面仍按预先定义的候选输入集收集数据。")


def show_cross_sectional_input_guidance(features):
    with st.expander("📐 必读：所需指标及计算/获取方法", expanded=False):
        st.info(
            "模型 E 的全部体成分指标应来自同一次检测；不要混用不同日期或不同设备的分段数据。"
            "比例类指标请特别留意页面要求的是0–1小数还是百分数点。"
            "优先读取设备导出的同名指标；必须手算时使用未圆整的原始分量。"
            "腰围、臀围及问卷项沿用原研究体检/问卷口径。"
        )
        for category in CATEGORY_ORDER:
            category_features = [
                name for name in features if FEATURE_DICT[name]["cat"] == category
            ]
            if not category_features:
                continue
            st.markdown(f"**{category}**")
            for feature_name in category_features:
                guidance = FEATURE_GUIDANCE.get(feature_name, "按体检或检测报告原值填写。")
                st.markdown(f"- **{display_name(feature_name)}**：{guidance}")


def show_follow_up_definition():
    st.markdown(
        """
        本研究的随访终点为受试者自基线起 **3 年（36个月）内**首次发生代谢异常。
        在血压、血糖和血脂三类代谢组分中出现 **两类及以上** 异常，即判定为终点发生。

        - **血压异常**：非同日两次血压达到或超过 130/85 mmHg，或出现高血压诊断、降压用药或相应自报信息。
        - **糖代谢异常**：空腹血糖 ≥5.6 mmol/L，或出现糖尿病前期/2型糖尿病诊断、降糖用药或相应自报信息。
        - **血脂异常**：TG ≥1.7 mmol/L、HDL-C 男性 <1.0 mmol/L/女性 <1.3 mmol/L、LDL-C ≥3.4 mmol/L、TC ≥5.2 mmol/L，或出现血脂异常诊断、降脂用药或相应自报信息。
        """
    )


def show_cross_sectional_definition():
    st.markdown(
        """
        横断面模型的因变量直接采用工作簿中的 **“代谢异常”**（0=否，1=是）。
        在本次数据中，它与下列四项异常的累计数 ≥2 完全一致：

        - 高血压/血压异常
        - 糖尿病/糖代谢异常
        - 甘油三酯异常
        - HDL-C 异常

        因此，本模型输出的是 **当前存在代谢异常的估计概率**，不是未来 3 年发病风险。
        """
    )


def show_follow_up_result(probability):
    if probability < 0.30:
        st.success("**低风险**：该受试者未来 3 年内发生代谢异常的模型估计风险较低。")
    elif probability < 0.60:
        st.warning("**中等风险**：该受试者未来 3 年内发生代谢异常的模型估计风险处于中间范围。")
    else:
        st.error("**高风险**：该受试者未来 3 年内发生代谢异常的模型估计风险较高。")
    st.progress(float(np.clip(probability, 0, 1)), text=f"未来 3 年发病概率：{probability:.2%}")
    st.info(
        "低（<30%）、中（30%–<60%）、高（≥60%）为网页展示分层，并非统一的临床决策阈值；"
        "请结合实际诊疗规范解释。"
    )


def show_cross_sectional_result(probability, threshold, metadata):
    is_positive = probability >= threshold
    if is_positive:
        st.error(f"**筛查阳性倾向**：估计概率达到模型判别阈值（{threshold:.1%}）。")
    else:
        st.success(f"**筛查阴性倾向**：估计概率低于模型判别阈值（{threshold:.1%}）。")
    st.progress(float(np.clip(probability, 0, 1)), text=f"当前代谢异常估计概率：{probability:.2%}")

    evaluation = metadata.get("evaluation", {})
    metrics = evaluation.get("metrics", {})
    if metrics:
        with st.expander("查看该横断面模型的内部验证表现", expanded=False):
            cols = st.columns(4)
            cols[0].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
            cols[1].metric("灵敏度", f"{metrics['sensitivity']:.1%}")
            cols[2].metric("特异度", f"{metrics['specificity']:.1%}")
            cols[3].metric("Brier", f"{metrics['brier_score']:.3f}")
            st.caption(
                f"固定 20% 内部测试集（n={evaluation.get('test_n', '—')}）；"
                "阈值由训练集 10 折折外预测的最大 Youden J 确定。"
            )
    st.info(
        "该结果用于研究性横断面筛查，不能替代诊断。模型目前仅完成单一数据集内部验证，"
        "临床使用前仍需独立外部验证与校准。"
    )


def warn_about_extrapolation(processed_data, metadata):
    reference = metadata.get("input_reference", {})
    outside = []
    for feature_name, value in processed_data.items():
        feature_reference = reference.get(feature_name, {})
        if feature_reference.get("type") != "numeric":
            continue
        lower = feature_reference.get("percentile_1")
        upper = feature_reference.get("percentile_99")
        if lower is not None and upper is not None and not lower <= value <= upper:
            outside.append(f"{FEATURE_DICT[feature_name]['label'] or feature_name}（训练数据1%–99%：{lower:.3g}–{upper:.3g}）")
    if outside:
        st.warning("以下输入超出训练数据的常见范围，预测可能属于外推：" + "；".join(outside))


st.sidebar.header("📂 模型选择与配置")
selected_model_name = st.sidebar.selectbox("请选择要使用的预测模型：", list(MODEL_REGISTRY))
selected_config = MODEL_REGISTRY[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.info(f"正在使用：\n**{selected_model_name}**\n\n{selected_config['description']}")
with st.sidebar.expander("ℹ️ 查看模型类型说明", expanded=False):
    st.markdown(
        "**随访模型 A–D**：估计未来3年风险。  \n"
        "**横断面模型 E**：人口学/生活方式、BMI、WHR和体成分。  \n"
        "**横断面模型 F**：仅人口学/生活方式、BMI和WHR。  \n"
        "各模型独立运行，并非概率融合。"
    )

st.title("🩺 代谢异常在线预测系统")
if selected_config["endpoint"] == "follow_up":
    st.markdown("当前选择的是 **随访预测模型**：评估受试者未来 3 年内发生代谢异常的概率。")
else:
    st.markdown("当前选择的是 **横断面判别模型**：评估受试者当前存在代谢异常的概率。")
    st.warning(
        "数据质控待确认：工作簿中的“HDL异常”标签与修正后的 HDL-C 数值不一致。"
        "当前横断面模型严格按您指定的“代谢异常”列训练；临床使用或投稿前请先确认标签是否已随 HDL 修正而重算。"
    )

with st.expander("📚 查看本模型的结局定义", expanded=False):
    if selected_config["endpoint"] == "follow_up":
        show_follow_up_definition()
    else:
        show_cross_sectional_definition()

if selected_config["endpoint"] == "cross_sectional":
    show_cross_sectional_model_overview(selected_config)
    show_cross_sectional_input_guidance(selected_config["features"])

features = selected_config["features"]
unknown_features = [name for name in features if name not in FEATURE_DICT]
if unknown_features:
    st.error(f"页面缺少字段定义：{unknown_features}")
    st.stop()

categories = sorted(
    {FEATURE_DICT[name]["cat"] for name in features},
    key=lambda category: CATEGORY_ORDER.index(category),
)
input_data = {}

with st.form("prediction_form"):
    for category in categories:
        st.markdown(f"### {category}")
        category_features = [name for name in features if FEATURE_DICT[name]["cat"] == category]
        col1, col2 = st.columns(2)
        for index, feature_name in enumerate(category_features):
            container = col1 if index % 2 == 0 else col2
            info = FEATURE_DICT[feature_name]
            with container:
                if feature_name in OPTIONS_MAP:
                    choices = OPTIONS_MAP[feature_name]
                    selected = st.selectbox(
                        display_label(feature_name),
                        options=list(choices),
                        help=FEATURE_GUIDANCE.get(feature_name),
                        key=f"{selected_model_name}:{feature_name}",
                    )
                    input_data[feature_name] = choices[selected]
                else:
                    input_data[feature_name] = st.number_input(
                        display_label(feature_name),
                        min_value=0.0,
                        max_value=info["max"],
                        value=float(info["def"]),
                        step=float(10 ** -info["decimals"]),
                        format=f"%.{info['decimals']}f",
                        help=FEATURE_GUIDANCE.get(feature_name),
                        key=f"{selected_model_name}:{feature_name}",
                    )
        st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🚀 点击进行预测", use_container_width=True)

if submitted:
    model_path = APP_DIR / selected_config["file"]
    if not model_path.exists():
        st.error(f"找不到模型文件：{model_path.name}")
        st.stop()

    try:
        bundle = load_model_bundle(selected_config["file"])
        model = bundle["model"]
        processed_data = {
            name: value * FEATURE_DICT[name].get("model_scale", 1.0)
            for name, value in input_data.items()
        }
        model_input = pd.DataFrame([processed_data], columns=features)

        if selected_config["endpoint"] == "cross_sectional":
            warn_about_extrapolation(processed_data, bundle.get("metadata", {}))

        expected_features = list(getattr(model, "feature_names_in_", features))
        if expected_features != features:
            raise ValueError(
                "页面字段与模型训练字段不一致："
                f"页面={features}，模型={expected_features}"
            )

        probabilities = model.predict_proba(model_input)[0]
        classes = list(model.classes_)
        positive_index = classes.index(1)
        probability = float(probabilities[positive_index])

        st.markdown("---")
        st.subheader("📊 预测结果")
        if selected_config["endpoint"] == "follow_up":
            show_follow_up_result(probability)
        else:
            threshold = float(bundle.get("threshold") or 0.5)
            show_cross_sectional_result(probability, threshold, bundle.get("metadata", {}))
    except Exception as error:
        st.error(f"模型运行出错：{error}")
