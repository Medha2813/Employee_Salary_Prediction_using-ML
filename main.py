# Employee Salary Prediction using Machine Learning (Linear Regression)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load Dataset
df = pd.read_csv("Dataset09-Employee-salary-prediction.csv")  # Make sure this file is in the same folder

# 2. Clean column names
df.columns = ['Age', 'Gender', 'Degree', 'Job_Title', 'Experience_Year', 'Salary']

# 3. Drop missing values
df.dropna(inplace=True)

# 4. Encode categorical columns
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
df['Degree_Encoded'] = le.fit_transform(df['Degree'])
df['Job_Title_Encoded'] = le.fit_transform(df['Job_Title'])

# 5. Scale numerical features
scaler = StandardScaler()
df['Age_Scaled'] = scaler.fit_transform(df[['Age']])
df['Experience_Scaled'] = scaler.fit_transform(df[['Experience_Year']])

# 6. Define features and target
X = df[['Age_Scaled', 'Gender_Encoded', 'Degree_Encoded', 'Job_Title_Encoded', 'Experience_Scaled']]
y = df['Salary']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)

# 10. Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("🔍 Model Evaluation:")
print(f"R2 Score (Accuracy): {r2*100:.2f}%")
print(f"Mean Absolute Error: ₹{mae:.2f}")
print(f"Mean Squared Error: ₹{mse:.2f}")
print(f"Root Mean Squared Error: ₹{rmse:.2f}")

# 11. Coefficients
print("\n📊 Model Coefficients:", model.coef_)
print("📈 Intercept:", model.intercept_)

# 12. Actual vs Predicted comparison
results = pd.DataFrame({'Actual Salary': y_test, 'Predicted Salary': y_pred})
results['Error'] = results['Actual Salary'] - results['Predicted Salary']
results['Absolute Error'] = np.abs(results['Error'])
print("\nComparison:")
print(results.head())

# 13. Plot actual vs predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.title("Actual vs Predicted Salary")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.grid(True)
plt.show()

# 14. Predict for new employee (example)
# Example input: Age=49, Gender=Female, Degree=PhD, Job_Title=Director, Experience=15
gender_encoded = le.transform(['Female'])[0]
degree_encoded = le.transform(['PhD'])[0]
job_title_encoded = le.transform(['Director'])[0]
age_scaled = scaler.transform([[49]])[0][0]
exp_scaled = scaler.transform([[15]])[0][0]

new_data = np.array([[age_scaled, gender_encoded, degree_encoded, job_title_encoded, exp_scaled]])
predicted_salary = model.predict(new_data)
print(f"\n💰 Predicted Salary for custom input: ₹{predicted_salary[0]:,.2f}")
