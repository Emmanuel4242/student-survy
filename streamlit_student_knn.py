import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.title("🎓 Student Performance Predictor (KNN)")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Stats survey.csv")

    # Convert GPA to numeric
    df["GPA_2023"] = pd.to_numeric(
        df["Your 2023 academic year average/GPA in % (Ignore if you are 2024 1st year student)"], 
        errors="coerce"
    )

    # Create performance level
    def classify_performance(score):
        if score < 60:
            return "low"
        elif score < 75:
            return "medium"
        else:
            return "high"

    df["performance level"] = df["GPA_2023"].apply(classify_performance)

    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = [
        "Your Sex?",
        "What faculty does your degree fall under?",
        "Were you on scholarship/bursary in 2023?",
        "How often do you go out partying/socialising during the week? ",
        "On a night out, how many alcoholic drinks do you consume?",
    ]

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

df = load_data()

# Features and label
X = df[[
    "Your Sex?",
    "What faculty does your degree fall under?",
    "Were you on scholarship/bursary in 2023?",
    "How often do you go out partying/socialising during the week? ",
    "On a night out, how many alcoholic drinks do you consume?"
]]
y = df["performance level"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Sidebar inputs
st.sidebar.header("Input Student Features")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
faculty = st.sidebar.selectbox("Faculty", sorted(df["What faculty does your degree fall under?"].unique()))
scholarship = st.sidebar.selectbox("Scholarship in 2023?", ["Yes", "No"])
partying = st.sidebar.slider("Party Frequency (times/week)", 0, 7, 1)
drinks = st.sidebar.slider("Drinks per Night Out", 0, 20, 1)

# Manual mapping to match training encoding
sex_map = {"Male": 0, "Female": 1}
faculty_map = {name: idx for idx, name in enumerate(sorted(df["What faculty does your degree fall under?"].unique()))}
scholarship_map = {"Yes": 1, "No": 0}

sex_encoded = sex_map[sex]
faculty_encoded = faculty_map[faculty]
scholarship_encoded = scholarship_map[scholarship]

# Format input
input_data = pd.DataFrame([[
    sex_encoded,
    faculty_encoded,
    scholarship_encoded,
    partying,
    drinks
]], columns=X.columns)

input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Performance Level"):
    prediction = knn.predict(input_scaled)[0]
    st.success(f"📊 Predicted Performance Level: **{prediction.upper()}**")
else:
    st.info("⬅️ Enter student details and click Predict")
