import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- 1. RESEARCH DATA CORE ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}
df_ml = pd.DataFrame(dry_data)
X, y = df_ml[['Speed', 'Feed', 'DOC']], df_ml['Temp']

# --- 2. AI MODEL VALIDATION ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
y_pred_all = rf_model.predict(X)
mape = mean_absolute_percentage_error(y, y_pred_all) * 100
accuracy_pct = 100 - mape

# --- 3. UI CONFIG ---
st.set_page_config(page_title="Inconel Research Center", layout="wide")

# Custom Professional Styling
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #0d1117; color: #8b949e; text-align: center; padding: 10px; border-top: 1px solid #30363d; z-index: 100; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.title("🛡️ Inconel 718: Thermal Precision Interface")
st.markdown(f"Advanced Predictive Analytics | Developed by **Mohammed Faheem M S**")
st.divider()

# --- 5. SIDEBAR & LIVE CALCULATION ---
st.sidebar.header("🕹️ Experimental Parameters")
dia = st.sidebar.number_input("Workpiece Diameter (mm)", value=25.0, format="%.4f")
in_speed = st.sidebar.number_input("Cutting Speed (m/min)", value=60.0, format="%.4f")
in_feed = st.sidebar.number_input("Feed Rate (mm/rev)", value=0.1, format="%.4f")
in_doc = st.sidebar.number_input("Depth of Cut (mm)", value=0.5, format="%.4f")

# Real-time Prediction
calc_rpm = (1000 * in_speed) / (math.pi * dia)
live_pred = rf_model.predict([[in_speed, in_feed, in_doc]])[0]

# --- 6. INSTRUMENTATION GAUGES ---
c1, c2 = st.columns(2)
with c1:
    fig_rpm = go.Figure(go.Indicator(mode="gauge+number", value=calc_rpm, 
        number={'valueformat': ".2f", 'font': {'size': 40, 'color': 'white'}},
        title={'text': "SPINDLE SPEED (RPM)", 'font': {'size': 18, 'color': '#58a6ff'}},
        gauge={'axis': {'range': [0, 4000], 'tickcolor': "white"}, 'bar': {'color': "#58a6ff"}}))
    fig_rpm.update_layout(height=350, margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rpm, use_container_width=True)

with c2:
    fig_temp = go.Figure(go.Indicator(mode="gauge+number", value=live_pred, 
        number={'valueformat': ".2f", 'font': {'size': 40, 'color': 'white'}},
        title={'text': "PREDICTED TEMPERATURE (°C)", 'font': {'size': 18, 'color': '#ff7b72'}},
        gauge={'axis': {'range': [0, 1300], 'tickcolor': "white"}, 'bar': {'color': "#ff7b72"}}))
    fig_temp.update_layout(height=350, margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_temp, use_container_width=True)

# --- 7. KEY PERFORMANCE INDICATORS (KPIs) ---
st.subheader("📊 Statistical Reliability")
k1, k2, k3 = st.columns(3)
k1.metric("Model Confidence (Accuracy)", f"{accuracy_pct:.2f}%")
k2.metric("Relative Error (MAPE)", f"{mape:.2f}%")
k3.metric("Live Prediction Variable", f"{live_pred:.2f} °C")

st.divider()

# --- 8. RESEARCH VALIDATION GRAPH ---
st.subheader("📈 Regression Analysis: Actual vs. Predicted")
fig_parity = go.Figure()
# Identity Line
fig_parity.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], 
                                mode='lines', name='Ideal Fit', line=dict(color='#8b949e', dash='dash')))
# Actual Data Points
fig_parity.add_trace(go.Scatter(x=y, y=y_pred_all, mode='markers', name='AI Observations', 
                                marker=dict(color='#ff7b72', size=10, opacity=0.7, line=dict(width=1, color='white'))))

fig_parity.update_layout(
    template="plotly_dark", height=500,
    xaxis_title="Experimental Result (FLIR Camera) °C",
    yaxis_title="AI Model Prediction °C",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_parity, use_container_width=True)

# --- 9. FOOTER ---
st.markdown(f'<div class="footer">Developed by <b>Mohammed Faheem M S</b> | Inconel 718 Research Hub © 2026</div>', unsafe_allow_html=True)
