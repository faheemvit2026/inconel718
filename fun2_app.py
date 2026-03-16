import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# --- 1. DATA PREPARATION ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}
df_ml = pd.DataFrame(dry_data)
features = ['Speed', 'Feed', 'DOC']
X, y = df_ml[features], df_ml['Temp']

# Train Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
lr_model = LinearRegression().fit(X, y)

# --- 2. LAYOUT ---
st.set_page_config(page_title="Inconel AI Model Lab", layout="wide")
st.title("🛡️ Inconel 718: Model Variance Visualization")

# --- 3. INPUT SIDEBAR ---
st.sidebar.header("🕹️ Parameters")
dia = st.sidebar.number_input("Diameter (mm)", value=25.0, format="%.4f")
in_speed = st.sidebar.number_input("Speed Vc (m/min)", value=60.0, format="%.4f")
in_feed = st.sidebar.number_input("Feed f (mm/rev)", value=0.1, format="%.4f")
in_doc = st.sidebar.number_input("DOC ap (mm)", value=0.5, format="%.4f")

# --- 4. CALCULATIONS ---
current_input = [[in_speed, in_feed, in_doc]]
rf_pred = rf_model.predict(current_input)[0]
lr_pred = lr_model.predict(current_input)[0]
var_pct = (abs(rf_pred - lr_pred) / rf_pred) * 100

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("AI Prediction", f"{rf_pred:.2f} °C")
m2.metric("Linear Baseline", f"{lr_pred:.2f} °C")
m3.metric("Variance", f"{var_pct:.2f} %")

# --- 5. VARIANCE PLOT ---
st.subheader("📊 Comparative Accuracy Analysis")

# Generate Plot
fig, ax = plt.subplots(figsize=(10, 4))
y_sorted = np.sort(y)
y_rf_all = rf_model.predict(X)
y_lr_all = lr_model.predict(X)
sorted_idx = np.argsort(y)

ax.scatter(range(len(y)), y_sorted, color='gray', alpha=0.5, label='Actual Trials')
ax.plot(range(len(y)), y_rf_all[sorted_idx], color='#ff9900', label='AI (RF)', linewidth=2)
ax.plot(range(len(y)), y_lr_all[sorted_idx], color='cyan', linestyle='--', label='Linear Trend')
ax.set_ylabel("Temperature (°C)")
ax.set_xlabel("Experimental Trials (Sorted)")
ax.legend()
ax.grid(True, alpha=0.2)

st.pyplot(fig)

st.info("💡 The gap between the dashed cyan line and the orange line represents the 12.94% non-linearity you discovered.")
