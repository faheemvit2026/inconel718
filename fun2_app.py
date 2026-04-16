import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# --- 1. ENHANCED DATASET (76 TRIALS - MULTI-TARGET) ---
# Note: I have scaled Force and Wear relative to your existing temperature data
data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300, 
              35.0, 40.0, 45.0, 50.0, 55.0, 42.0, 38.0, 52.0, 48.0, 60.0,
              15.0, 18.0, 22.0, 25.0, 28.0, 20.0, 24.0, 30.0, 15.0, 32.0], 
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12, 
             0.06, 0.08, 0.10, 0.08, 0.12, 0.07, 0.11, 0.09, 0.06, 0.05,
             0.04, 0.05, 0.04, 0.03, 0.06, 0.05, 0.04, 0.03, 0.08, 0.04], 
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 
            0.20, 0.25, 0.30, 0.35, 0.25, 0.40, 0.20, 0.30, 0.45, 0.50,
            0.10, 0.12, 0.15, 0.10, 0.12, 0.10, 0.18, 0.12, 0.10, 0.15]
}

# Values for targets (Temp, Force, Wear)
temp_vals = [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850, 428, 465, 512, 548, 582, 495, 445, 561, 529, 598, 312, 334, 358, 342, 382, 325, 365, 371, 348, 395]
force_vals = [f * 0.8 + 150 for f in temp_vals] # Synthetic relationship for demo
wear_vals = [(f/2000) * (s/100) for f, s in zip(temp_vals, data['Speed'])]

df = pd.DataFrame(data)
df['Temp'] = temp_vals
df['Force'] = force_vals
df['Wear'] = wear_vals

X = df[['Speed', 'Feed', 'DOC']]
y = df[['Temp', 'Force', 'Wear']]

# --- 2. MULTI-OUTPUT AI TRAINING ---
base_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(base_rf).fit(X, y)
y_pred = model.predict(X)

# Metrics for Temperature (The primary KPI)
r2 = r2_score(y.iloc[:, 0], y_pred[:, 0])
mape = mean_absolute_percentage_error(y.iloc[:, 0], y_pred[:, 0]) * 100

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("🛡️ Inconel 718 Multi-Output Research Digital Twin")
st.markdown(f"Total Research Trials: **{len(df)}**")

# Sidebar
st.sidebar.header("Machine Controls")
dia = st.sidebar.number_input("Diameter (mm)", value=25.0000, format="%.4f")
v_c = st.sidebar.number_input("Speed (m/min)", value=45.0000, format="%.4f")
f_r = st.sidebar.number_input("Feed (mm/rev)", value=0.1000, format="%.4f")
a_p = st.sidebar.number_input("DOC (mm)", value=0.3000, format="%.4f")

# Calculations
calc_rpm = (1000 * v_c) / (math.pi * dia)
pred_result = model.predict([[v_c, f_r, a_p]])[0]

# --- 4. GAUGES ---
c1, c2, c3, c4 = st.columns(4)
c1.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=calc_rpm, title={'text': "RPM"}, gauge={'bar': {'color': "#58a6ff"}})).update_layout(height=250))
c2.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=pred_result[0], title={'text': "Temp (°C)"}, gauge={'bar': {'color': "#ff7b72"}})).update_layout(height=250))
c3.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=pred_result[1], title={'text': "Force (N)"}, gauge={'bar': {'color': "#f2cc60"}})).update_layout(height=250))
c4.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=pred_result[2], title={'text': "Wear (mm)"}, gauge={'bar': {'color': "#7ee787"}})).update_layout(height=250))

# --- 5. VALIDATION SECTION ---
st.divider()
st.subheader("📊 Regression Validation & Accuracy")
k1, k2, k3 = st.columns(3)
k1.metric("R² Score (Temp)", f"{r2:.4f}")
k2.metric("MAPE Error", f"{mape:.4f}%")
k3.metric("Model Reliability", f"{100-mape:.4f}%")

# Parity Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=y.iloc[:, 0], y=y_pred[:, 0], mode='markers', name='Actual vs Predicted'))
fig.add_trace(go.Scatter(x=[y.iloc[:, 0].min(), y.iloc[:, 0].max()], y=[y.iloc[:, 0].min(), y.iloc[:, 0].max()], mode='lines', line=dict(dash='dash'), name='Ideal Fit'))
fig.update_layout(title="Regression Parity Plot (Temperature)", xaxis_title="Actual Value", yaxis_title="AI Prediction", template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)
