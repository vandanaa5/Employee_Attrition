import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load files safely using absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'employee_attrition_model.pkl'))
feature_columns = joblib.load(os.path.join(BASE_DIR, 'feature_columns.pkl'))

st.title("Employee Attrition Prediction")
st.markdown("Enter the employee details to predict if they are likely to leave the company.")

# Sidebar input
st.sidebar.header("Employee Details")

def get_user_input():
    inputs = {}
    inputs['Age'] = st.sidebar.number_input("Age", 18, 65, 30)
    inputs['MonthlyIncome'] = st.sidebar.number_input("MonthlyIncome", 1000, 20000, 5000)
    inputs['JobSatisfaction'] = st.sidebar.selectbox("JobSatisfaction", [1,2,3,4])
    inputs['OverTime'] = st.sidebar.selectbox("OverTime", ["Yes","No"])
    inputs['DistanceFromHome'] = st.sidebar.number_input("DistanceFromHome", 0, 50, 10)

    # Convert to DataFrame
    df = pd.DataFrame([inputs])

    # FIX: manual encoding instead of LabelEncoder
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

    # Ensure correct feature order
    data = {}
    for feat in feature_columns:
        if feat in df.columns:
            data[feat] = df[feat][0]
        else:
            data[feat] = 0

    return pd.DataFrame([data])

user_input = get_user_input()

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[0][1]

    if prediction[0] == 1:
        st.error("The employee is likely to leave the company.")
    else:
        st.success("The employee is likely to stay with the company.")

    st.info(f"Prediction Probability: {probability:.2f}")