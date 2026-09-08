from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="代谢异常风险预测系统", page_icon="🩺", layout="centered")


# 模型中的字段名均使用训练时保存的原始名称，展示名称单独放在 FEATURE_DICT 中。
MODEL_REGISTRY = {
    "模型A（体成分+实验室）": {
        "file": "lab_model_a.pkl",
        "features": [
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
        ],
        "input_features": [
            "体重指数",
            "收缩压",
            "下肢肌肉比率",
            "细胞外液总量/身体总水分",
            "上肢肌肉比率",
            "下肢脂肪百分比",
            "糖化血红蛋白",
            "总胆固醇",
            "甘油三酯",
            "葡萄糖",
        ],
        "input_limits": {
            "体重指数": {"min": 10.0, "max": 70.0},
            "下肢肌肉比率": {"min": 0.0001, "max": 1.0},
            "细胞外液总量/身体总水分": {"min": 0.001, "max": 1.0},
            "上肢肌肉比率": {"min": 0.0001, "max": 1.0},
            "下肢脂肪百分比": {"min": 0.0001, "max": 1.0},
        },
        "description": "使用体格、体成分和实验室检查指标评估代谢异常风险；TyG 指数由葡萄糖和甘油三酯自动计算。",
    },
    "模型E（人口学+体成分）": {
        "file": "cross_sectional_bodycomp_lasso.pkl",
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
        "description": "使用人口学、生活方式、BMI、腰臀比和无创体成分指标评估代谢异常风险。",
    },
    "模型F（人口学基础）": {
        "file": "cross_sectional_basic_lasso.pkl",
        "features": ["性别", "年龄", "运动频率", "吸烟史", "饮酒史", "体重指数", "腰臀比"],
        "description": "仅使用人口学、生活方式、BMI 和腰臀比评估代谢异常风险。",
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
    min_value=0.0,
    max_value=None,
):
    return {
        "cat": category,
        "unit": unit,
        "def": default,
        "label": label,
        "decimals": decimals,
        "model_scale": model_scale,
        "min": min_value,
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
    "收缩压": feature(PHYSICAL, "mmHg", 120.0, decimals=0, min_value=80.0, max_value=250.0),
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
    "甘油三酯": feature(LAB, "mmol/L", 1.5, min_value=0.01, max_value=50.0),
    "葡萄糖": feature(LAB, "mmol/L", 5.5, min_value=0.01, max_value=50.0),
    "糖化血红蛋白": feature(LAB, "%", 5.5, min_value=3.0, max_value=20.0),
    "TyG 指数": feature(LAB, "", 8.5),
    "尿酸": feature(LAB, "μmol/L", 300.0),
    "门冬氨酸氨基转移酶": feature(LAB, "U/L", 20.0),
    "尿素": feature(LAB, "mmol/L", 5.0),
    "嗜酸性粒细胞百分比": feature(LAB, "%", 2.0),
    "肌酐": feature(LAB, "μmol/L", 70.0),
    "血红蛋白": feature(LAB, "g/L", 135.0),
    "总胆固醇": feature(LAB, "mmol/L", 4.5, min_value=0.01, max_value=20.0),
    "嗜碱性粒细胞百分比": feature(LAB, "%", 0.5),
    "丙氨酸氨基转移酶": feature(LAB, "U/L", 20.0),
    "脂蛋白α测定": feature(LAB, "mg/L", 150.0),
    "红细胞计数": feature(LAB, "10^12/L", 4.5),
    "白细胞计数": feature(LAB, "10^9/L", 6.0),
    "γ-谷氨酰转移酶": feature(LAB, "U/L", 25.0),
    "白蛋白": feature(LAB, "g/L", 45.0),
}


# 所有体成分指标应来自同一次检测，避免混用不同日期的报告。
FEATURE_GUIDANCE = {
    "性别": "按本研究编码选择：男性=1，女性=0。",
    "年龄": "以检查日期减出生日期计算周岁，填写完整岁数。",
    "运动频率": "按平均每周运动次数分组：<1次、1–2次、3–5次或>5次。",
    "吸烟史": "从未吸烟选择“无”；当前吸烟或既往吸烟/已戒烟均选择“有”。",
    "饮酒史": "按研究问卷填写：否认饮酒=0；任何有饮酒记录（包括应酬性饮酒）=1。",
    "体重指数": "BMI = 体重(kg) ÷ 身高(m)²。例如70 kg、1.75 m，BMI=22.86 kg/m²。模型训练使用临床身高、体重计算的“体重指数”，不是InBody的“身体质量指数”。",
    "腰臀比": "WHR = 腰围 ÷ 臀围。两者使用相同单位，按原表训练口径保留2位小数；例如腰围85 cm、臀围95 cm，填写0.89。模型训练使用临床腰围、臀围计算的“腰臀比”，不是“腰臀比INBODY”。",
    "收缩压": "按规范静息后测量收缩压，填写检测报告中的数值，单位为mmHg。",
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
    "糖化血红蛋白": "填写检验报告中的糖化血红蛋白百分数，例如5.6%填写5.6。",
    "总胆固醇": "填写检验报告中的总胆固醇数值，单位为mmol/L。",
    "甘油三酯": "填写空腹采血检验报告中的甘油三酯数值，单位为mmol/L。",
    "葡萄糖": (
        "填写空腹采血检验报告中的葡萄糖数值，单位为mmol/L。网页会自动计算 "
        "TyG 指数：ln[(甘油三酯×88.5)×(葡萄糖×18)÷2]，无需手填TyG。"
    ),
}


TRIGLYCERIDE_MMOL_L_TO_MG_DL = 88.5
GLUCOSE_MMOL_L_TO_MG_DL = 18.0


def calculate_tyg(triglyceride_mmol_l, glucose_mmol_l):
    """Calculate TyG from fasting triglyceride and glucose values in mmol/L."""
    if triglyceride_mmol_l <= 0 or glucose_mmol_l <= 0:
        raise ValueError("甘油三酯和葡萄糖必须大于0，才能计算TyG指数。")
    return float(
        np.log(
            (
                triglyceride_mmol_l
                * TRIGLYCERIDE_MMOL_L_TO_MG_DL
                * glucose_mmol_l
                * GLUCOSE_MMOL_L_TO_MG_DL
            )
            / 2.0
        )
    )


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


def show_input_guidance(features):
    with st.expander("📐 必读：所需指标及计算/获取方法", expanded=False):
        has_body_composition = any(
            FEATURE_DICT[name]["cat"] == BODY for name in features
        )
        has_laboratory = any(FEATURE_DICT[name]["cat"] == LAB for name in features)
        if has_body_composition:
            st.info(
                "全部体成分指标应来自同一次检测；不要混用不同日期或不同设备的分段数据。"
                "比例类指标请特别留意页面要求的是0–1小数还是百分数点。"
                "优先读取设备导出的同名指标；必须手算时使用未圆整的原始分量。"
                "腰围、臀围及问卷项沿用原研究体检/问卷口径。"
            )
        else:
            st.info(
                "人口学、生活方式、腰围和臀围等指标应沿用原研究体检/问卷口径；"
                "BMI与腰臀比按下方公式计算。"
            )
        if has_laboratory:
            st.info(
                "实验室指标应尽量来自同一次空腹采血，并严格按页面标注的单位填写。"
                "TyG指数由网页根据甘油三酯和葡萄糖自动计算。"
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


def show_result_guidance():
    st.markdown(
        """
        - **代谢异常风险**：百分比越高，表示模型估算的代谢异常风险越高。
        - **风险分类**：估计概率达到或超过该模型的固定判别阈值时显示“风险较高”，低于阈值时显示“风险较低”。
        - **使用范围**：结果仅供风险评估参考，不能替代病史询问、体格检查、实验室检查或医生诊断。
        """
    )


def show_risk_result(probability, threshold):
    is_positive = probability >= threshold
    if is_positive:
        st.error(
            f"**代谢异常风险较高**：模型估计概率 {probability:.2%} "
            f"≥ 模型固定判别阈值 {threshold:.1%}。"
        )
    else:
        st.success(
            f"**代谢异常风险较低**：模型估计概率 {probability:.2%} "
            f"< 模型固定判别阈值 {threshold:.1%}。"
        )
    st.progress(float(np.clip(probability, 0, 1)), text=f"代谢异常风险：{probability:.2%}")
    st.info("本结果为模型估计，仅供风险评估参考，不能替代临床诊断。")


def warn_about_extrapolation(processed_data, metadata, auxiliary_data=None):
    reference = dict(metadata.get("input_reference", {}))
    derived_input = metadata.get("derived_input", {})
    auxiliary_reference = metadata.get("auxiliary_input_reference") or derived_input.get(
        "auxiliary_input_reference", {}
    )
    auxiliary_data = auxiliary_data or {}
    if auxiliary_reference:
        if "percentile_1" in auxiliary_reference:
            for feature_name in auxiliary_data:
                reference[feature_name] = {
                    "type": "numeric",
                    **auxiliary_reference,
                }
        else:
            for feature_name, feature_reference in auxiliary_reference.items():
                reference[feature_name] = {
                    "type": "numeric",
                    **feature_reference,
                }

    values_to_check = {**processed_data, **auxiliary_data}
    outside = []
    for feature_name, value in values_to_check.items():
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
with st.sidebar.expander("ℹ️ 查看模型适用信息", expanded=False):
    st.markdown(
        "**模型 A**：体格、体成分和实验室检查指标。  \n"
        "**模型 E**：人口学、生活方式、BMI、WHR和体成分指标。  \n"
        "**模型 F**：人口学、生活方式、BMI和WHR。  \n"
        "各模型独立运行，并非概率融合。"
    )

st.title("🩺 代谢异常风险在线预测系统")
st.markdown("请选择模型并填写相应指标，系统将估算代谢异常风险。")

with st.expander("💡 结果如何理解", expanded=False):
    show_result_guidance()

input_features = selected_config.get("input_features", selected_config["features"])
show_input_guidance(input_features)

features = selected_config["features"]
unknown_features = [name for name in {*features, *input_features} if name not in FEATURE_DICT]
if unknown_features:
    st.error(f"页面缺少字段定义：{unknown_features}")
    st.stop()

categories = sorted(
    {FEATURE_DICT[name]["cat"] for name in input_features},
    key=lambda category: CATEGORY_ORDER.index(category),
)
input_data = {}

with st.form("prediction_form"):
    for category in categories:
        st.markdown(f"### {category}")
        category_features = [
            name for name in input_features if FEATURE_DICT[name]["cat"] == category
        ]
        col1, col2 = st.columns(2)
        for index, feature_name in enumerate(category_features):
            container = col1 if index % 2 == 0 else col2
            info = FEATURE_DICT[feature_name]
            input_limits = selected_config.get("input_limits", {}).get(feature_name, {})
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
                        min_value=input_limits.get("min", info["min"]),
                        max_value=input_limits.get("max", info["max"]),
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
        st.error("模型暂时无法加载，请联系维护人员。")
        st.stop()

    try:
        bundle = load_model_bundle(selected_config["file"])
        model = bundle["model"]
        processed_data = {
            name: input_data[name] * FEATURE_DICT[name].get("model_scale", 1.0)
            for name in features
            if name != "TyG 指数"
        }
        if "TyG 指数" in features:
            processed_data["TyG 指数"] = calculate_tyg(
                input_data["甘油三酯"],
                input_data["葡萄糖"],
            )
            st.session_state["last_computed_tyg"] = processed_data["TyG 指数"]
        model_input = pd.DataFrame([processed_data], columns=features)

        auxiliary_data = (
            {"葡萄糖": input_data["葡萄糖"]} if "葡萄糖" in input_data else None
        )
        warn_about_extrapolation(
            processed_data,
            bundle.get("metadata", {}),
            auxiliary_data,
        )

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
        threshold = float(bundle.get("threshold") or 0.5)
        show_risk_result(probability, threshold)
    except Exception as error:
        st.error(f"模型运行出错：{error}")
