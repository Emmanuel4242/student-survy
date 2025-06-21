import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.title("🎓 Student Performance Predictor (KNN)")

# Load and preprocess dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Stats survey.csv")
        
        # Convert GPA to numeric
        gpa_col = "Your 2023 academic year average/GPA in % (Ignore if you are 2024 1st year student)"
        df["GPA_2023"] = pd.to_numeric(df[gpa_col], errors="coerce")
        
        # Create performance level
        def classify_performance(score):
            if pd.isna(score):
                return "unknown"
            if score < 60:
                return "low"
            elif score < 75:
                return "medium"
            return "high"

        df["performance level"] = df["GPA_2023"].apply(classify_performance)
        df = df[df["performance level"] != "unknown"]  # Remove rows with unknown GPA
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = [
            "Your Sex?",
            "What faculty does your degree fall under?",
            "Were you on scholarship/bursary in 2023?",
            "How often do you go out partying/socialising during the week?",
            "On a night out, how many alcoholic drinks do you consume?",
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data available. Please check your data file.")
    st.stop()

# Features and label
features = [
    "Your Sex?",
    "What faculty does your degree fall under?",
    "Were you on scholarship/bursary in 2023?",
    "How often do you go out partying/socialising during the week?",
    "On a night out, how many alcoholic drinks do you consume?"
]
X = df[features]
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
faculty_options = sorted(df["What faculty does your degree fall under?"].unique())
faculty = st.sidebar.selectbox("Faculty", faculty_options)
scholarship = st.sidebar.selectbox("Scholarship in 2023?", ["Yes", "No"])
partying = st.sidebar.slider("Party Frequency (times/week)", 0, 7, 1)
drinks = st.sidebar.slider("Drinks per Night Out", 0, 20, 1)

# Manual mapping to match training encoding
sex_map = {"Male": 0, "Female": 1}
faculty_map = {name: idx for idx, name in enumerate(faculty_options)}
scholarship_map = {"Yes": 1, "No": 0}

# Encode inputs
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
]], columns=features)

# Prediction button
if st.button("Predict Performance Level"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = knn.predict(input_scaled)[0]
        st.success(f"📊 Predicted Performance Level: {prediction.upper()}")
        
        # Show input details
        st.subheader("Input Details")
        st.json({
            "Sex": sex,
            "Faculty": faculty,
            "Scholarship": scholarship,
            "Party Frequency (times/week)": partying,
            "Drinks per Night Out": drinks
        })
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
else:
    st.info("Please enter student details in the sidebar and click the prediction button")
