import streamlit as st
import pandas as pd
import joblib

st.title("Student Survey KNN Classifier")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("knn_model.pkl")

model = load_model()

# Sidebar input
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 15, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
study_time = st.sidebar.slider("Study Time (hours/week)", 1, 20)

# Convert gender to number (you can adjust this based on your training data)
gender_num = 0 if gender == "Male" else 1

# Format input
input_data = pd.DataFrame([[age, gender_num, study_time]], columns=["age", "gender", "study_time"])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
