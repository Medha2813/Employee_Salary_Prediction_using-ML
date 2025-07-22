import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("Dataset09-Employee-salary-prediction.csv")
df.columns = ['Age', 'Gender', 'Degree', 'Job_Title', 'Experience_Year', 'Salary']
df.dropna(inplace=True)

# Manual encodings for Gender and Degree
gender_map = {'Male': 1, 'Female': 0}
degree_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}

# Label encode job title
job_encoder = LabelEncoder()
df['Job_Title_Encoded'] = job_encoder.fit_transform(df['Job_Title'])

# Encode Gender and Degree manually
df['Gender_Encoded'] = df['Gender'].map(gender_map)
df['Degree_Encoded'] = df['Degree'].map(degree_map)

# Scale Age and Experience
scaler = StandardScaler()
df['Age_Scaled'] = scaler.fit_transform(df[['Age']])
df['Experience_Scaled'] = scaler.fit_transform(df[['Experience_Year']])

# Prepare features and target
X = df[['Age_Scaled', 'Gender_Encoded', 'Degree_Encoded', 'Job_Title_Encoded', 'Experience_Scaled']]
y = df['Salary']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))

# Streamlit UI
st.title("💼 Employee Salary Predictor")

age = st.slider("Select Age", 18, 65, 25)
gender = st.selectbox("Select Gender", ["Male", "Female"])
degree = st.selectbox("Select Degree", ["Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Select Job Title", sorted(df['Job_Title'].unique()))
experience = st.slider("Years of Experience", 0, 40, 1)

if st.button("Predict Salary"):
    # Encode gender and degree
    gender_encoded = gender_map.get(gender, 0)
    degree_encoded = degree_map.get(degree, 0)

    # Encode job title safely
    if job_title in job_encoder.classes_:
        job_encoded = job_encoder.transform([job_title])[0]
    else:
        job_encoded = 0  # fallback

    # Scale age and experience
    age_scaled = scaler.transform([[age]])[0][0]
    exp_scaled = scaler.transform([[experience]])[0][0]

    input_data = np.array([[age_scaled, gender_encoded, degree_encoded, job_encoded, exp_scaled]])
    predicted_salary = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹{predicted_salary:,.2f}")

st.caption(f"Model Accuracy (R² Score): {r2 * 100:.2f}%")
