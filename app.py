import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. 页面基本设置 ---
st.set_page_config(page_title="代谢异常风险预测系统", page_icon="🩺", layout="centered")
st.title("🩺 代谢异常风险在线预测系统")
st.markdown("本系统基于多模型融合算法构建，用于评估受试者**未来 3 年内**发生代谢异常的风险。请填写下方患者临床指标。")

# 💡 新增1：代谢异常的临床定义与判定标准（折叠面板）
with st.expander("📚 点击查看：本研究中【代谢异常】的定义与判定标准", expanded=False):
    st.markdown("""
    本研究的**主要终点事件**定义为：受试者自基线访视起 **3年（36个月）内**，首次发生新发代谢异常。
    
    其**判定标准**为：在血压、血糖和血脂三类核心代谢组分中出现**两类及以上**异常。具体指标如下：
    
    * **(1) 血压异常**：若非同日两次测量的血压均达到或超过 130/85 mmHg，或明确出现高血压诊断并伴有降压药物处方，则判定为事件发生；若受试者自述被医生告知血压升高，或已规律使用降压药物，同样视为血压异常事件。
    * **(2) 糖代谢异常**：若检测提示空腹血糖（FPG）≥ 5.6 mmol/L，或记录有新的“糖尿病前期”或“2型糖尿病”诊断并伴随降糖药物处方，则判定为糖代谢异常；若受试者自述被提示血糖升高，或已开始规律服用降糖药物，也计入糖代谢异常事件。
    * **(3) 血脂异常**：若血脂谱任一指标出现异常，即甘油三酯（TG）≥ 1.7 mmol/L、HDL-C（男性＜1.0 mmol/L，女性＜1.3 mmol/L）、低密度脂蛋白胆固醇（LDL-C）≥ 3.4 mmol/L 或总胆固醇（TC）≥ 5.2 mmol/L，或出现新的血脂异常诊断及降脂药物处方，则判定为血脂异常；若受试者报告近期体检提示血脂异常，或自述已规律服用降脂药物，则同样认定为事件。
    """)

# --- 2. 模型定义与字典映射 ---
MODEL_FEATURES = {
    "模型A（全变量模型）": ['下肢肌肉比率', 'TyG 指数', '糖化血红蛋白', '体重指数', '总胆固醇', '尿素', '细胞外液总量/身体总水分', 'γ-谷氨酰转移酶', '甘油三酯', '白蛋白', '嗜酸性粒细胞百分比', '收缩压', '嗜碱性粒细胞百分比', '血红蛋白', '上肢肌肉比率', '下肢脂肪比率'],
    "模型B（体成分模型）": ['年龄', '体重指数', '腰臀比', '体脂肪', '体脂百分比', '上肢肌肉比率', '躯干肌肉量比率', '下肢肌肉比率', '躯干脂肪比率', '下肢脂肪比率', '细胞外液总量/身体总水分', '吸烟史', '身体总水分', '身体总水分/去脂体重'],
    "模型C（临床常规模型）": ['甘油三酯', '糖化血红蛋白', '体重指数', 'TyG 指数', '尿酸', '门冬氨酸氨基转移酶', '年龄', '尿素', '嗜酸性粒细胞百分比', '肌酐', '血红蛋白', '总胆固醇', '嗜碱性粒细胞百分比', '丙氨酸氨基转移酶', '脂蛋白α测定', '红细胞计数', '舒张压', '白细胞计数', '收缩压', 'γ-谷氨酰转移酶', '白蛋白', '腰臀比'],
    "模型D（基准模型）": ['性别', '年龄', '运动频率', '吸烟史', '饮酒史', '体重指数', '腰臀比']
}

MODEL_FILES = {
    "模型A（全变量模型）": "lasso_model.pkl",
    "模型B（体成分模型）": "svm_model.pkl",
    "模型C（临床常规模型）": "xgb_model.pkl",
    "模型D（基准模型）": "lr_model.pkl"
}

OPTIONS_MAP = {
    '性别': {"1 (男性)": 1, "0 (女性)": 0},
    '吸烟史': {"0 (无吸烟史)": 0, "1 (有吸烟史)": 1},
    '饮酒史': {"0 (无饮酒史)": 0, "1 (有饮酒史)": 1},
    '运动频率': {
        "0：极少 (＜1次/周)": 0, 
        "1：偶尔 (1-2次/周)": 1, 
        "2：规律 (3-5次/周)": 2, 
        "3：经常 (＞5次/周)": 3
    }
}

CATEGORY_ORDER = [
    '👤 基本人口学及生活方式',
    '📏 体格检查指标',
    '⚖️ 体成分指标',
    '🩸 实验室指标'
]

FEATURE_DICT = {
    '年龄': {'cat': '👤 基本人口学及生活方式', 'unit': '岁', 'def': 50.0},
    '性别': {'cat': '👤 基本人口学及生活方式', 'unit': '', 'def': None},
    '运动频率': {'cat': '👤 基本人口学及生活方式', 'unit': '', 'def': None},
    '吸烟史': {'cat': '👤 基本人口学及生活方式', 'unit': '', 'def': None},
    '饮酒史': {'cat': '👤 基本人口学及生活方式', 'unit': '', 'def': None},
    
    '体重指数': {'cat': '📏 体格检查指标', 'unit': 'kg/m²', 'def': 24.0},
    '腰臀比': {'cat': '📏 体格检查指标', 'unit': '', 'def': 0.85},
    '收缩压': {'cat': '📏 体格检查指标', 'unit': 'mmHg', 'def': 120.0},
    '舒张压': {'cat': '📏 体格检查指标', 'unit': 'mmHg', 'def': 80.0},
    
    '体脂肪': {'cat': '⚖️ 体成分指标', 'unit': 'kg', 'def': 15.0},
    '体脂百分比': {'cat': '⚖️ 体成分指标', 'unit': '%', 'def': 25.0},
    '上肢肌肉比率': {'cat': '⚖️ 体成分指标', 'unit': '上肢肌肉量/全身肌肉量', 'def': 0.11},
    '躯干肌肉量比率': {'cat': '⚖️ 体成分指标', 'unit': '躯干肌肉量/全身肌肉量', 'def': 0.47},
    '下肢肌肉比率': {'cat': '⚖️ 体成分指标', 'unit': '下肢肌肉量/全身肌肉量', 'def': 0.33},
    '躯干脂肪比率': {'cat': '⚖️ 体成分指标', 'unit': '躯干脂肪量/全身体脂肪量', 'def': 0.44},
    '下肢脂肪比率': {'cat': '⚖️ 体成分指标', 'unit': '下肢脂肪量/全身体脂肪量', 'def': 0.33},
    '细胞外液总量/身体总水分': {'cat': '⚖️ 体成分指标', 'unit': '', 'def': 0.38},
    '身体总水分': {'cat': '⚖️ 体成分指标', 'unit': 'kg', 'def': 37.5},
    '身体总水分/去脂体重': {'cat': '⚖️ 体成分指标', 'unit': '比值', 'def': 0.73},
    
    '甘油三酯': {'cat': '🩸 实验室指标', 'unit': 'mmol/L', 'def': 1.5},
    '糖化血红蛋白': {'cat': '🩸 实验室指标', 'unit': '%', 'def': 5.5},
    'TyG 指数': {'cat': '🩸 实验室指标', 'unit': '', 'def': 8.5},
    '尿酸': {'cat': '🩸 实验室指标', 'unit': 'μmol/L', 'def': 300.0},
    '门冬氨酸氨基转移酶': {'cat': '🩸 实验室指标', 'unit': 'U/L', 'def': 20.0},
    '尿素': {'cat': '🩸 实验室指标', 'unit': 'mmol/L', 'def': 5.0},
    '嗜酸性粒细胞百分比': {'cat': '🩸 实验室指标', 'unit': '%', 'def': 2.0},
    '肌酐': {'cat': '🩸 实验室指标', 'unit': 'μmol/L', 'def': 70.0},
    '血红蛋白': {'cat': '🩸 实验室指标', 'unit': 'g/L', 'def': 135.0},
    '总胆固醇': {'cat': '🩸 实验室指标', 'unit': 'mmol/L', 'def': 4.5},
    '嗜碱性粒细胞百分比': {'cat': '🩸 实验室指标', 'unit': '%', 'def': 0.5},
    '丙氨酸氨基转移酶': {'cat': '🩸 实验室指标', 'unit': 'U/L', 'def': 20.0},
    '脂蛋白α测定': {'cat': '🩸 实验室指标', 'unit': 'mg/L', 'def': 150.0},
    '红细胞计数': {'cat': '🩸 实验室指标', 'unit': '10^12/L', 'def': 4.5},
    '白细胞计数': {'cat': '🩸 实验室指标', 'unit': '10^9/L', 'def': 6.0},
    'γ-谷氨酰转移酶': {'cat': '🩸 实验室指标', 'unit': 'U/L', 'def': 25.0},
    '白蛋白': {'cat': '🩸 实验室指标', 'unit': 'g/L', 'def': 45.0},
}

# --- 3. 侧边栏：选择模型与简介 ---
st.sidebar.header("📂 模型选择与配置")
selected_model_name = st.sidebar.selectbox("请选择要使用的预测模型：", list(MODEL_FEATURES.keys()))

st.sidebar.markdown("---")
st.sidebar.info(f"正在使用：\n**{selected_model_name}**")

with st.sidebar.expander("ℹ️ 查看各模型适用场景说明", expanded=True):
    st.markdown("""
    * **模型A（全变量）**：纳入全部多维度指标，用于评估全量信息条件下的预测性能上限。
    * **模型B（体成分）**：包含基础指标与无创体成分数据，无需抽血化验。适合在基层医疗或仅具备体成分仪的场景下进行初筛。
    * **模型C（临床常规）**：包含基础指标与常规抽血生化数据，不依赖体成分仪。贴近传统临床常规诊疗场景。
    * **模型D（基准）**：仅需人口学、生活方式及基础体格指标（BMI/腰臀比），作为最基础的对照参考。
    """)

# --- 4. 主界面：动态、分类生成输入表单 ---
features = MODEL_FEATURES[selected_model_name]
input_data = {}

unsorted_categories = list(set([FEATURE_DICT[f]['cat'] for f in features if f in FEATURE_DICT]))
categories_in_model = sorted(unsorted_categories, key=lambda x: CATEGORY_ORDER.index(x) if x in CATEGORY_ORDER else 99)

with st.form("prediction_form"):
    for category in categories_in_model:
        st.markdown(f"### {category}")
        cat_features = [f for f in features if f in FEATURE_DICT and FEATURE_DICT[f]['cat'] == category]
        
        col1, col2 = st.columns(2)
        for i, feature in enumerate(cat_features):
            col = col1 if i % 2 == 0 else col2
            unit_text = f" ({FEATURE_DICT[feature]['unit']})" if FEATURE_DICT[feature]['unit'] else ""
            display_label = f"{feature}{unit_text}"
            
            with col:
                if feature in OPTIONS_MAP:
                    option_dict = OPTIONS_MAP[feature]
                    display_choice = st.selectbox(display_label, options=list(option_dict.keys()))
                    input_data[feature] = option_dict[display_choice]
                else:
                    default_val = FEATURE_DICT[feature]['def']
                    input_data[feature] = st.number_input(display_label, value=default_val, format="%f")
        st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🚀 点击进行风险预测", use_container_width=True)

# --- 5. 执行预测逻辑 ---
if submitted:
    processed_data = input_data.copy()
    
    if '身体总水分/去脂体重' in processed_data:
        processed_data['身体总水分/去脂体重'] = processed_data['身体总水分/去脂体重'] * 100

    df = pd.DataFrame([processed_data])
    
    RENAME_FOR_MODEL = {
        '躯干脂肪比率': '躯干脂肪百分比',
        '下肢脂肪比率': '下肢脂肪百分比'
    }
    df = df.rename(columns=RENAME_FOR_MODEL)
    
    original_feature_order = MODEL_FEATURES[selected_model_name]
    aligned_feature_order = [RENAME_FOR_MODEL.get(f, f) for f in original_feature_order]
    df = df[aligned_feature_order]
    
    try:
        model_path = MODEL_FILES[selected_model_name]
        if not os.path.exists(model_path):
            st.error(f"❌ 找不到模型文件: {model_path}")
        else:
            model = joblib.load(model_path)
            pred_class = model.predict(df)[0]
            pred_proba = float(model.predict_proba(df)[0][1])
            
            st.markdown("---")
            st.subheader("📊 风险预测评估结果")
            
            if pred_proba < 0.3:
                st.success(f"**✅ 低风险**：该受试者未来 3 年内发生代谢异常的风险较低。")
                st.progress(pred_proba, text=f"3年内发病概率：{pred_proba:.2%}")
            elif pred_proba < 0.6:
                st.warning(f"**⚠️ 中等风险**：该受试者未来 3 年内发生代谢异常的风险处于临界状态，建议生活方式干预。")
                st.progress(pred_proba, text=f"3年内发病概率：{pred_proba:.2%}")
            else:
                st.error(f"**🚨 高风险**：该受试者未来 3 年内发生代谢异常的可能性很大，建议临床密切关注。")
                st.progress(pred_proba, text=f"3年内发病概率：{pred_proba:.2%}")

            # 💡 新增2：风险评估结果的详细说明框
            st.info("""
            **💡 评估结果说明：**
            * **指标意义**：上方百分比代表该受试者在**未来 3 年内**发生上述“代谢异常”事件的综合概率。
            * **计算逻辑**：由后台机器学习算法综合您填写的基线数据，经过特征权重计算与非线性映射，输出的一个 0%~100% 的发病倾向得分。
            * **风险分层依据**：
                * **🟢 低风险（< 30%）**：模型研判其 3 年内发病概率较低，建议继续保持良好生活方式。
                * **🟡 中等风险（30% - 60%）**：模型研判受试者处于发病临界区间或已具备部分危险因素，建议开始生活方式干预并定期随访。
                * **🔴 高风险（≥ 60%）**：具备强烈的代谢异常进展倾向，建议由临床医生进一步评估，必要时进行医疗干预。
            * *(注：以上阈值为本系统设定参考，具体临床决策请结合实际诊疗规范)*
            """)

    except Exception as e:
        st.error(f"❌ 模型运行出错：{str(e)}")
