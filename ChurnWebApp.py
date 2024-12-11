import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load pre-trained model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Preprocessing function
def preprocess_input(data):
    # Map Yes/No columns to 1/0
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                      'PaperlessBilling']
    for column in yes_no_columns:
        data[column] = data[column].map({'Yes': 1, 'No': 0})

    # Gender mapping
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

    # One-hot encoding for categorical columns
    data = pd.get_dummies(data, columns=['Contract', 'PaymentMethod', 'InternetService'], drop_first=True)

    # Standardize numerical columns
    scaler = StandardScaler()
    data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        data[['tenure', 'MonthlyCharges', 'TotalCharges']])

    return data


# Streamlit app layout
st.title("Customer Churn Prediction App")
st.write("Enter the customer details below:")

# Input fields for features
gender = st.selectbox("Gender", options=["Male", "Female"])
partner = st.selectbox("Partner", options=["Yes", "No"])
dependents = st.selectbox("Dependents", options=["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", options=["Yes", "No"])
internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", options=["Yes", "No"])
online_backup = st.selectbox("Online Backup", options=["Yes", "No"])
device_protection = st.selectbox("Device Protection", options=["Yes", "No"])
tech_support = st.selectbox("Tech Support", options=["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", options=["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", options=["Yes", "No"])
contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
payment_method = st.selectbox("Payment Method",
                              options=["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                       "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=0.1)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'gender': [gender],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Preprocess input
    preprocessed_data = preprocess_input(input_data)

    # Predict
    prediction = model.predict(preprocessed_data)
    prediction_probability = model.predict_proba(preprocessed_data)

    # Display result
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.write(f"Churn Probability: {prediction_probability[0][1]:.2f}")
