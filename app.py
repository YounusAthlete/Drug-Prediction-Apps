import streamlit as st
import numpy as np
import pandas as pd
import joblib  # or pickle
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load model and encoder saved from training (encoder no longer needed for input transform)
model = joblib.load("decision_tree_model.pkl")
# encoder = joblib.load("ordinal_encoder.pkl")  # Not used for input encoding now

st.title("💊 Drug Prediction App")
st.write("Enter patient details to predict the recommended drug.")

# Mapping dictionaries for categorical inputs
sex_map = {"M": 0.0, "F": 1.0}
bp_map = {"LOW": 0.0, "NORMAL": 1.0, "HIGH": 2.0}
cholesterol_map = {"NORMAL": 0.0, "HIGH": 1.0}

# User inputs including Age
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", ["M", "F"])
bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
cholesterol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
na_to_k = st.number_input("Na_to_K Ratio", min_value=0.0, max_value=100.0, value=15.0)

if st.button("Predict Drug"):
    # Map categorical inputs to numeric codes
    sex_code = sex_map[sex]
    bp_code = bp_map[bp]
    cholesterol_code = cholesterol_map[cholesterol]

    # Create DataFrame for model input
    input_df = pd.DataFrame([[age, sex_code, bp_code, cholesterol_code, na_to_k]],
                            columns=["Age", "Sex", "BP", "Cholesterol", "Na_to_K"])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Drug: **{prediction}**")


