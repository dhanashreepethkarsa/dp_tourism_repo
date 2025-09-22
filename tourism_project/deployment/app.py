import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained tourism model
# Replace with your Hugging Face repo ID and model file name
model_path = hf_hub_download(
    repo_id="DhanashreeP/Tourism-mlops-prediction-FE", 
    filename="best_tourism_prediction_model_v1.joblib"
)

model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Product Purchase Prediction")
st.write("""
This app predicts whether a customer is likely to purchase the tourism package (**ProdTaken**).
Please provide the customer details below to get a prediction.
""")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
duration = st.number_input("Duration of Travel (days)", min_value=1, max_value=30, value=5, step=1)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=30000, step=1000)
family_members = st.number_input("Number of Family Members", min_value=0, max_value=10, value=2, step=1)

gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widow"])
occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Small Business", "Others"])
mode_of_transport = st.selectbox("Preferred Mode of Transport", ["Car", "Bus", "Air", "Train"])
passport = st.selectbox("Passport", ["Yes", "No"])
prod_taken = None  # This will be predicted

# Prepare input DataFrame (align columns with training features)
input_data = pd.DataFrame([{
    "Age": age,
    "Duration": duration,
    "MonthlyIncome": monthly_income,
    "NumberOfFamilyMembers": family_members,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "Occupation": occupation,
    "ModeOfTransport": mode_of_transport,
    "Passport": passport
}])

# Predict
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase (ProdTaken=1)" if prediction == 1 else "Will Not Purchase (ProdTaken=0)"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
