import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesIQ — Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0ede6;
}

.stApp {
    background: #0d0d0d;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #f5e642, #f0a500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #888;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Section labels */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #f5e642;
    margin-bottom: 0.8rem;
    margin-top: 1.8rem;
}

/* Cards */
.card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result boxes */
.result-safe {
    background: linear-gradient(135deg, #0a2e1a, #0d3d22);
    border: 1px solid #1a6b3a;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-risk {
    background: linear-gradient(135deg, #2e0a0a, #3d0d0d);
    border: 1px solid #8b1a1a;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.result-sub {
    font-size: 0.95rem;
    color: #bbb;
    margin-top: 0.5rem;
}
.prob-badge {
    display: inline-block;
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    margin: 0.5rem 0;
}

/* Slider tweaks */
.stSlider > div > div > div {
    background: #f5e642 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #f5e642, #f0a500);
    color: #0d0d0d;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.85;
}

/* Info tip */
.tip {
    background: #1a1a1a;
    border-left: 3px solid #f5e642;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #999;
    margin-top: 0.5rem;
}

/* Divider */
hr { border-color: #2a2a2a; }

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load or Train Model ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Try loading saved model first
    if os.path.exists("best_diabetes_model.pkl") and os.path.exists("scaler.pkl"):
        model = joblib.load("best_diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler

    # Otherwise train a fresh one from the dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    df = pd.read_csv(url, names=cols)

    for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, scaler


# ── App Layout ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🩺 DiabetesIQ</h1>
    <p>AI-powered diabetes risk prediction · Enter patient vitals below</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 10, 90, 35)
    pregnancies = st.slider("Pregnancies", 0, 17, 1)
    glucose = st.slider("Glucose Level (mg/dL)", 50, 250, 110)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 20, 130, 72)

with col2:
    bmi = st.slider("BMI", 10.0, 70.0, 28.0, step=0.1)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 25)
    insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 80)
    dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.50, 0.47, step=0.01)

st.markdown('<div class="tip">💡 Diabetes Pedigree Function reflects family history. Higher = stronger genetic risk.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("⚡ PREDICT DIABETES RISK"):
    with st.spinner("Analyzing patient data..."):
        model, scaler = load_model()

        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        prob_pct = f"{probability * 100:.1f}%"

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-risk">
            <div class="result-title">⚠️ High Diabetes Risk</div>
            <div class="prob-badge" style="color:#ff6b6b">{prob_pct}</div>
            <div class="result-sub">probability of diabetes detected</div>
            <div class="result-sub" style="margin-top:1rem">
                Please consult a healthcare professional for further testing.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-title">✅ Low Diabetes Risk</div>
            <div class="prob-badge" style="color:#51cf66">{prob_pct}</div>
            <div class="result-sub">probability of diabetes detected</div>
            <div class="result-sub" style="margin-top:1rem">
                Maintain a healthy lifestyle. Regular check-ups are still recommended.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Feature breakdown
    st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Feature": ["Age", "Pregnancies", "Glucose", "Blood Pressure",
                    "BMI", "Skin Thickness", "Insulin", "Pedigree Fn"],
        "Your Value": [age, pregnancies, glucose, blood_pressure,
                       bmi, skin_thickness, insulin, dpf],
        "Normal Range": ["—", "0–17", "70–100 mg/dL", "60–80 mm Hg",
                         "18.5–24.9", "10–35 mm", "16–166 mu/ml", "< 0.5"]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#444; font-size:0.78rem;'>
DiabetesIQ · ML Model: Random Forest trained on Pima Indians Diabetes Dataset<br>
⚠️ For educational purposes only. Not a substitute for medical advice.<br>
🧑‍💻 Created by Arunesh
</p>
""", unsafe_allow_html=True)
