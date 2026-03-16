import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# --- 1. DATA CORE (Validated Research Dataset) ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}
df = pd.DataFrame(dry_data)
X_feats = df[['Speed', 'Feed', 'DOC']]
y_actual = df['Temp']

# --- 2. AI TRAINING & STATISTICS ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_feats, y_actual)
y_pred_total = rf_model.predict(X_feats)

# Scientific Metrics
r2 = r2_score(y_actual, y_pred_total)
mse = mean_squared_error(y_actual, y_pred_total)
mape = mean_absolute_percentage_error(y_actual, y_pred_total) * 100
accuracy = 100 - mape

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="Inconel 718 Research", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #0d1117; color: #8b949e; text-align: center; padding: 10px; border-top: 1px solid #30363d; z-index: 100; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER & BRANDING ---
st.title("🛡️ Inconel 718: Thermal Analysis Center")
st.markdown(f"Advanced Machine Learning Predictive Model | Developed by **Mohammed Faheem M S**")
st.divider()

# --- 5. SIDEBAR PARAMETERS ---
st.sidebar.header("🕹️ Control Parameters")
dia = st.sidebar.number_input("Workpiece Diameter (mm)", value=25.0, format="%.2f")
v_c = st.sidebar.number_input("Cutting Speed (m/min)", value=60.0, format="%.2f")
f_r = st.sidebar.number_input("Feed Rate (mm/rev)", value=0.1, format="%.2f")
a_p = st.sidebar.number_input("Depth of Cut (mm)", value=0.5, format="%.2f")

# Live Calculations
calc_rpm = (1000 * v_c) / (math.pi * dia)
sim_temp = rf_model.predict([[v_c, f_r, a_p]])[0]

# --- 6. VISUAL GAUGES ---
col1, col2 = st.columns(2)
with col1:
    fig_rpm = go.Figure(go.Indicator(mode="gauge+number", value=calc_rpm, 
        number={'font': {'color': 'white'}}, title={'text': "SPINDLE RPM", 'font': {'color': '#58a6ff'}},
        gauge={'axis': {'range': [0, 4000]}, 'bar': {'color': "#58a6ff"}}))
    fig_rpm.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rpm, use_container_width=True)

with col2:
    fig_temp = go.Figure(go.Indicator(mode="gauge+number", value=sim_temp, 
        number={'font': {'color': 'white'}}, title={'text': "PREDICTED TEMP (°C)", 'font': {'color': '#ff7b72'}},
        gauge={'axis': {'range': [0, 1300]}, 'bar': {'color': "#ff7b72"}}))
    fig_temp.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_temp, use_container_width=True)

# --- 7. RESEARCH METRICS ---
st.subheader("📊 Statistical Validation")
k1, k2, k3, k4 = st.columns(4)
k1.metric("R² Score", f"{r2:.4f}")
k2.metric("Mean Squared Error", f"{mse:.2f}")
k3.metric("MAPE Error", f"{mape:.2f}%")
k4.metric("Model Accuracy", f"{accuracy:.2f}%")

st.divider()

# --- 8. REGRESSION PARITY PLOT ---
st.subheader("📈 Regression Parity: Actual vs. Predicted Temperature")
fig_parity = go.Figure()
# Identity line
fig_parity.add_trace(go.Scatter(x=[y_actual.min(), y_actual.max()], y=[y_actual.min(), y_actual.max()], 
                                mode='lines', name='Perfect Prediction', line=dict(color='#8b949e', dash='dash')))
# Actual Data
fig_parity.add_trace(go.Scatter(x=y_actual, y=y_pred_total, mode='markers', name='Experimental Data', 
                                marker=dict(color='#ff7b72', size=10, opacity=0.7, line=dict(width=1, color='white'))))

fig_parity.update_layout(
    template="plotly_dark", height=500,
    xaxis_title="Actual Temperature (FLIR Camera) °C",
    yaxis_title="AI Predicted Temperature °C",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_parity, use_container_width=True)

# --- 9. FOOTER ---
st.markdown(f'<div class="footer">Developed by <b>Mohammed Faheem M S</b> | Inconel 718 Research Hub © 2026</div>', unsafe_allow_html=True)
