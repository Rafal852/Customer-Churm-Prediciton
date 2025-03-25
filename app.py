import streamlit as st
import joblib
import numpy as np

try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please check the deployment files.")
    st.stop()


st.title("Churn Prediction App")
st.divider()
st.write("Please enter the values below and hit the prediction button to get a prediction.")


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

with col2:
    monthly_charge = st.number_input("Enter Monthly Charge", min_value=0, max_value=150, value=50)
    gender = st.selectbox("Enter Gender", ["Male", "Female"])

st.divider()


predict_button = st.button("Predict!")
st.divider()

if predict_button:
    
    gender_selected = 1 if gender == "Female" else 0
    X = [age, gender_selected, tenure, monthly_charge]
    
    X_array = scaler.transform([X])
    
    prediction = model.predict(X_array)[0]
    
    predicted_message = "Customer is likely to churn." if prediction == 1 else "Customer is unlikely to churn."
    
    st.success(f"Prediction: {predicted_message}")
else:
    st.info("Please enter values and click the Predict button.")
