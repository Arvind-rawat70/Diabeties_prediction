import streamlit as st
import joblib
import numpy as np

# Load the saved model pipeline
model = joblib.load('diabetes_model.pkl')

# Streamlit app
st.title("Diabetes Prediction App")

st.markdown("""
Enter the patient's medical data below to predict if they are diabetic.
""")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

# Prediction
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to be **Diabetic**.")
    else:
        st.success("✅ The person is likely **Not Diabetic**.")
