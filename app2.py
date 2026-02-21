# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("💳 Credit Risk Prediction App")

st.write("Enter customer details below:")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", value=50000)
loan_amount = st.number_input("Loan Amount", value=20000)
marital_status = st.selectbox("Marital Status", ["Single", "Married"])

if st.button("Predict"):

    new_data = {
        "Age": age,
        "Income": income,
        "Loan Amount": loan_amount,
        "Marital Status": marital_status
    }

    df = pd.DataFrame([new_data])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    if prediction == 0:
        result = "high risk "
    elif prediction == 1:
        result = "low risk "
    else:
        result = "low risk "

    st.success(f"Prediction: {result}")