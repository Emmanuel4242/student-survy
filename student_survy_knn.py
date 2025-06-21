# 1. Install & Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


df = pd.read_csv("/content/Stats survey.csv")
df.head()


# In[ ]:


# Create a performance label from GPA
df["GPA_2023"] = pd.to_numeric(df["Your 2023 academic year average/GPA in % (Ignore if you are 2024 1st year student)"], errors="coerce")

def classify_performance(score):
    if score < 60:
        return "low"
    elif score < 75:
        return "medium"
    else:
        return "high"

df["performance level"] = df["GPA_2023"].apply(classify_performance)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

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


# In[ ]:


X = df[[ "Your Sex?",
    "What faculty does your degree fall under?",
    "Were you on scholarship/bursary in 2023?",
    "How often do you go out partying/socialising during the week? ",
    "On a night out, how many alcoholic drinks do you consume?"]]
y = df["performance level"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Akurasi: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)


# In[ ]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Between Features in the Dataset")
plt.tight_layout()
plt.show()


# In[ ]:


# Calculate the average of two GPA columns (assuming they are numeric)
df["Matric_and_2023_avg"] = df[["Your Matric (grade 12) Average/ GPA (in %)",
                                "Your 2023 academic year average/GPA in % (Ignore if you are 2024 1st year student)"]].mean(axis=1)

# Function to determine pass/fail status (adjust threshold as needed)
def graduation_status(score):
    return "pass" if score >= 70 else "fail"

# Add a new column for graduation status
df["graduation status"] = df["Matric_and_2023_avg"].apply(graduation_status)

# Save the updated DataFrame to a new CSV file
df.to_csv("Stats_survey_with_graduation.csv", index=False)

# Display the first few rows of the updated DataFrame
df.head()


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support

labels = ["low", "medium", "high"]
metrics = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
scores_df = pd.DataFrame(metrics[:3], index=["Precision", "Recall", "F1-score"], columns=labels).T

scores_df.plot(kind="bar", figsize=(8,6))
plt.title("Classification Report by Class")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.grid(axis='y')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[ ]:


df["graduation status"].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90,
    colors=['#66b3ff', '#ff9999']
)

plt.title("Graduation Status Distribution")
plt.ylabel("")
plt.show()


# In[ ]:


df["performance level"].value_counts().plot(
    kind='bar',
    color=['#ff9999', '#66b3ff', '#99ff99']
)

plt.title("Performance Level Distribution")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()


# In[ ]:


# Convert relevant columns to numeric, coercing errors to NaN
cols_to_numeric = [
    "Additional amount of studying (in hrs) per week",
    "How often do you go out partying/socialising during the week? ",
    "On a night out, how many alcoholic drinks do you consume?"
]
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Graduation
graduation_means = df.groupby("graduation status")[cols_to_numeric].mean()
graduation_means.plot(kind='bar', figsize=(8,6))
plt.title("Average Score per Graduation Category")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# Performance level
performance_means = df.groupby("performance level")[cols_to_numeric].mean()
performance_means.plot(kind='bar', figsize=(8,6), color=["#FF9999", "#66B3FF", "#99FF99"])
plt.title("Average Score per Performance Level")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

