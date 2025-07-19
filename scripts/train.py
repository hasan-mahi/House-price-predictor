import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Ensure model folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
print("Dataset sample:")
print(df.head())

# Features & target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# Train models
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)


# Evaluation
def evaluate(model, name):
    pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name} — MSE: {mse:.4f}, R²: {r2:.4f}")
    return mse, r2


evaluate(lr, "Linear Regression")
evaluate(rf, "Random Forest")

# Save best model
joblib.dump(rf, "models/house_price_model.pkl")
print("Model saved as models/house_price_model.pkl")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf.predict(X_test_scaled), alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (Random Forest)")
plt.tight_layout()
plt.savefig("models/prediction_plot.png")
plt.close()

# Optional: open image in terminal environment
# os.system("xdg-open models/prediction_plot.png")  # For Linux
os.system("start models/prediction_plot.png")  # For Windows
# os.system("open models/prediction_plot.png")    # For macOS
