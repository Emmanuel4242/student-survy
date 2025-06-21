import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("knn_model.pkl")

model = load_model()

st.set_page_config(page_title="Student Study Prediction", page_icon="📘")
st.title("📘 Predict Student Outcome with KNN")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
study_time = st.slider("Study Time (hours per week)", 1, 20, 5)

# Encode gender
gender_num = 0 if gender == "Male" else 1

# Format input
input_data = pd.DataFrame([[gender_num, study_time]], columns=["gender", "study_time"])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Prediction: **{prediction}**")

    
