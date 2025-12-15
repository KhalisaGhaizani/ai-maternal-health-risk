import streamlit as st
import numpy as np
import pickle
import pandas as pd

# =========================
# Load trained model
# =========================
with open("maternal_risk_rf_best.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Maternal Health Risk Prediction")
st.write("Enter maternal health information to predict risk level.")

# =========================
# Input fields
# =========================
age = st.number_input("Age", min_value=10, max_value=60, value=25)

systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 120)
diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 140, 80)

blood_sugar = st.number_input("Blood Sugar (mmol/L)", 3.0, 20.0, 6.0, step=0.1)

body_temp = st.number_input("Body Temperature (°F)", 95.0, 105.0, 98.6, step=0.1)

heart_rate = st.number_input("Heart Rate (bpm)", 40, 150, 75)

# =========================
# Prediction button
# =========================
if st.button("Predict Risk"):
    # Order MUST match training data
    input_data = np.array([[
        age,
        systolic_bp,
        diastolic_bp,
        blood_sugar,
        body_temp,
        heart_rate
    ]])

    # Predict probabilities
    probs = model.predict_proba(input_data)[0]

    # Soft decision logic
    if probs[2] >= 0.35:
        risk = "HIGH RISK"
        color = "🔴"
    elif probs[1] >= 0.40:
        risk = "MID RISK"
        color = "🟠"
    else:
        risk = "LOW RISK"
        color = "🟢"

    # =========================
    # Output
    # =========================
    st.subheader("Prediction Result")
    st.markdown(f"### {color} **{risk}**")

    st.write("### Prediction Confidence")
    st.write(f"Low Risk:  {probs[0]*100:.2f}%")
    st.write(f"Mid Risk:  {probs[1]*100:.2f}%")
    st.write(f"High Risk: {probs[2]*100:.2f}%")
