import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- 1. RESEARCH DATA ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}
df_ml = pd.DataFrame(dry_data)
X, y = df_ml[['Speed', 'Feed', 'DOC']], df_ml['Temp']

# --- 2. MODEL & ERROR CALCULATION ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
y_pred_all = rf_model.predict(X)

# Calculate Errors
mape = mean_absolute_percentage_error(y, y_pred_all) * 100 # Error Percentage
mae = mean_absolute_error(y, y_pred_all) # Degree Error
accuracy_pct = 100 - mape

# --- 3. UI SETUP ---
st.set_page_config(page_title="Inconel Research | Mohammed Faheem M S", layout="wide")

st.markdown("""
    <style>
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #0e1117; color: white; text-align: center; padding: 10px; font-size: 14px; border-top: 1px solid #333; z-index: 100; }
    .metric-box { background-color: #161b22; border-radius: 10px; padding: 20px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718: Thermal Precision Center")
st.markdown("Developed by **Mohammed Faheem M S**")
st.divider()

# --- 4. ERROR METRICS SECTION ---
st.subheader("📉 Prediction Error Analysis")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Error Percentage (MAPE)", f"{mape:.2f}%", delta=f"-{mape:.2f}%", delta_color="inverse")
    st.caption("Lower is better. Represents average deviation from actual values.")

with c2:
    st.metric("Model Accuracy", f"{accuracy_pct:.2f}%")
    st.caption("Overall reliability of the AI model.")

with c3:
    st.metric("Mean Absolute Error", f"{mae:.2f} °C")
    st.caption("Average error in degrees Celsius.")

st.divider()

# --- 5. DATA TABLE (Actual vs Predicted) ---
st.subheader("📋 Experimental Validation Table")
comparison_df = pd.DataFrame({
    "Actual Temp (°C)": y,
    "AI Predicted Temp (°C)": y_pred_all,
    "Absolute Error (°C)": np.abs(y - y_pred_all),
    "Error (%)": (np.abs(y - y_pred_all) / y) * 100
})
st.dataframe(comparison_df.style.format("{:.4f}"), use_container_width=True)

# --- FOOTER ---
st.markdown(f'<div class="footer">Developed by <b>Mohammed Faheem M S</b></div>', unsafe_allow_html=True)
