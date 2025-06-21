import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.title("🎓 Student Performance Predictor (KNN)")

# Load and cache data
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
                return None
            if score < 60:
                return "low"
            elif score < 75:
                return "medium"
            return "high"

        df["performance_level"] = df["GPA_2023"].apply(classify_performance)
        df = df.dropna(subset=["performance_level"])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data available. Please check your data file.")
    st.stop()

# Prepare features and target
features = [
    "Your Sex?",
    "What faculty does your degree fall under?",
    "Were you on scholarship/bursary in 2023?",
    "How often do you go out partying/socialising during the week?",
    "On a night out, how many alcoholic drinks do you consume?"
]

# Encode categorical variables
le = LabelEncoder()
for col in features:
    df[col] = le.fit_transform(df[col].astype(str))

X = df[features]
y = df["performance_level"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train and cache model
@st.cache_resource
def train_model(X, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model

model = train_model(X_scaled, y)

# Sidebar inputs
st.sidebar.header("Input Student Features")

# Get unique values for selectboxes
sex_options = ["Male", "Female"]
faculty_options = sorted(df["What faculty does your degree fall under?"].unique())
scholarship_options = ["Yes", "No"]

sex = st.sidebar.selectbox("Sex", sex_options)
faculty = st.sidebar.selectbox("Faculty", faculty_options)
scholarship = st.sidebar.selectbox("Scholarship in 2023?", scholarship_options)
partying = st.sidebar.slider("Party Frequency (times/week)", 0, 7, 1)
drinks = st.sidebar.slider("Drinks per Night Out", 0, 20, 1)

# Prepare input data
input_dict = {
    "Your Sex?": sex,
    "What faculty does your degree fall under?": faculty,
    "Were you on scholarship/bursary in 2023?": scholarship,
    "How often do you go out partying/socialising during the week?": partying,
    "On a night out, how many alcoholic drinks do you consume?": drinks
}

input_df = pd.DataFrame([input_dict])

# Encode input data
for col in features:
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col].astype(str))

# Scale input
input_scaled = scaler.transform(input_df[features])

# Predict
if st.button("Predict Performance Level"):
    try:
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        
        st.subheader("Prediction Result")
        st.success(f"📊 Performance Level: **{prediction.upper()}**")
        
        st.subheader("Prediction Probabilities")
        proba_df = pd.DataFrame({
            "Performance Level": ["Low", "Medium", "High"],
            "Probability": proba
        })
        st.bar_chart(proba_df.set_index("Performance Level"))
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
else:
    st.info("⬅️ Input student details and click Predict")
