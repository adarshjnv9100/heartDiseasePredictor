from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Resolve model artifact paths
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "KNN_heart.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
expected_columns = joblib.load(BASE_DIR / "columns.pkl")

# Header
st.title("❤️ Heart Disease Prediction")
st.markdown(
    "Provide the patient's clinical indicators below to assess heart disease risk using a trained K-Nearest Neighbors (KNN) model."
)

st.divider()

# Input layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox(
        "Sex",
        ["M", "F"],
        format_func=lambda x: "Male" if x == "M" else "Female"
    )
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"],
        help="ATA: Atypical Angina | NAP: Non-Anginal Pain | TA: Typical Angina | ASY: Asymptomatic"
    )
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        format_func=lambda x: "Yes (> 120 mg/dL)" if x == 1 else "No (≤ 120 mg/dL)"
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        help="Normal: Normal | ST: ST-T wave abnormality | LVH: Left ventricular hypertrophy"
    )
    max_hr = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"],
        format_func=lambda x: "Yes" if x == "Y" else "No"
    )
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        help="Up: Upsloping | Flat: Flat | Down: Downsloping"
    )

st.divider()

# Prediction logic
if st.button("Predict Risk", type="primary", use_container_width=True):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    risk_score = probabilities[1] * 100

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease** (Estimated Risk: {risk_score:.1f}%)")
        st.warning("Recommendation: Please consult a qualified healthcare provider for thorough cardiovascular evaluation.")
    else:
        st.success(f"✅ **Low Risk of Heart Disease** (Estimated Risk: {risk_score:.1f}%)")
        st.info("Recommendation: Maintain a healthy lifestyle with balanced nutrition and routine physical activity.")
