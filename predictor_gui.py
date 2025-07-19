import sys
import os
import platform
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, messagebox

# Detect base directory depending on how the app runs
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS  # PyInstaller's temp dir
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    messagebox.showerror(
        "Error", f"Model or scaler file not found at:\n{MODEL_PATH}\n{SCALER_PATH}"
    )
    sys.exit(1)

# Predefined US locations with lat/lon
US_LOCATIONS = {
    "New York, NY": (40.7128, -74.0060),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Chicago, IL": (41.8781, -87.6298),
    "Houston, TX": (29.7604, -95.3698),
    "Phoenix, AZ": (33.4484, -112.0740),
    "Philadelphia, PA": (39.9526, -75.1652),
    "San Antonio, TX": (29.4241, -98.4936),
    "San Diego, CA": (32.7157, -117.1611),
    "Dallas, TX": (32.7767, -96.7970),
    "San Jose, CA": (37.3382, -121.8863),
}

# Friendly feature labels for user input
feature_labels = {
    "MedInc": "Median Income (in 10k USD)",
    "HouseAge": "House Age (years)",
    "AveRooms": "Average Rooms per House",
    "AveBedrms": "Average Bedrooms per House",
    "Population": "Neighborhood Population",
    "AveOccup": "Average Occupants per House",
}

features = list(feature_labels.keys())

# Create main window
root = tk.Tk()
root.title("🏠 House Price Predictor")
root.geometry("420x560")
root.configure(bg="#f7f7f7")
root.resizable(False, False)

# Optional: Handle DPI scaling on Windows
if platform.system() == "Windows":
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

# Styling
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=("Segoe UI", 10), background="#f7f7f7")
style.configure("TButton", font=("Segoe UI", 10, "bold"))

# UI Frame
frame = ttk.Frame(root, padding=20)
frame.pack(fill="both", expand=True)

ttk.Label(
    frame, text="Enter House Features Below:", font=("Segoe UI", 12, "bold")
).pack(pady=(0, 20))

entries = {}

for feature_key in features:
    ttk.Label(frame, text=feature_labels[feature_key]).pack(anchor="w", pady=(5, 0))
    entry = ttk.Entry(frame, width=30)
    entry.pack(fill="x", pady=(0, 5))
    entries[feature_key] = entry

# Location dropdown for US cities (replaces Latitude and Longitude inputs)
ttk.Label(frame, text="Location (US Cities)").pack(anchor="w", pady=(5, 0))
location_var = tk.StringVar()
location_dropdown = ttk.Combobox(
    frame, textvariable=location_var, state="readonly", values=list(US_LOCATIONS.keys())
)
location_dropdown.pack(fill="x", pady=(0, 5))
location_dropdown.current(0)  # default first city selected


def predict_price():
    try:
        # Collect input values for features except lat/lon
        values = [float(entries[f].get()) for f in features]

        # Append lat/lon based on selected location
        lat, lon = US_LOCATIONS[location_var.get()]
        values.append(lat)
        values.append(lon)

        input_array = np.array(values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]
        price = prediction * 100000
        messagebox.showinfo("Predicted Price", f"Estimated House Price:\n${price:,.2f}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter all numeric values.")


ttk.Button(frame, text="Predict House Price", command=predict_price).pack(pady=(20, 0))

root.mainloop()
