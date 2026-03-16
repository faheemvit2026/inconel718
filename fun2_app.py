import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Inconel AI Precision Hub", layout="wide")

# Custom CSS for the footer and branding
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #333;
        z-index: 999;
    }
    .header-name {
        color: #ff9900;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER & BRANDING ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🛡️ Inconel 718: Thermal Command Center")
with col_h2:
    st.markdown("<br><p class='header-name'>Mohammed Faheem M S</p>", unsafe_allow_html=True)

# --- 4. INPUT SIDEBAR ---
st.sidebar.header("🕹️ Parameters")
dia = st.sidebar.number_input("Rod Diameter (mm)", value=25.0, format="%.4f")
in_speed = st.sidebar.number_input("Speed Vc (m/min)", value=60.0, format="%.4f")
in_feed = st.sidebar.number_input("Feed f (mm/rev)", value=0.1, format="%.4f")
in_doc = st.sidebar.number_input("DOC ap (mm)", value=0.5, format="%.4f")

# Calculations
calc_rpm = (1000 * in_speed) / (math.pi * dia)
rf_pred = rf_model.predict([[in_speed, in_feed, in_doc]])[0]
lr_pred = lr_model.predict([[in_speed, in_feed, in_doc]])[0]
variance = (abs(rf_pred - lr_pred) / rf_pred) * 100

# --- 5. ANALOGUE GAUGES ---
col_rpm, col_temp = st.columns(2)

with col_rpm:
    fig_rpm = go.Figure(go.Indicator(
        mode = "gauge+number", value = calc_rpm,
        title = {'text': "SPINDLE RPM", 'font': {'color': 'cyan'}},
        gauge = {'axis': {'range': [0, 4000]}, 'bar': {'color': "cyan"}}
    ))
    st.plotly_chart(fig_rpm, use_container_width=True)

with col_temp:
    fig_temp = go.Figure(go.Indicator(
        mode = "gauge+number", value = rf_pred,
        title = {'text': "AI TEMPERATURE (°C)", 'font': {'color': '#ff9900'}},
        gauge = {'axis': {'range': [0, 1300]}, 'bar': {'color': "#ff9900"}}
    ))
    st.plotly_chart(fig_temp, use_container_width=True)

# --- 6. MODEL VARIANCE GRAPH ---
st.divider()
st.subheader("📊 Variance Analysis (AI vs Linear Baseline)")

y_rf_all = rf_model.predict(X)
y_lr_all = lr_model.predict(X)
sorted_idx = np.argsort(y)

fig_var = go.Figure()
fig_var.add_trace(go.Scatter(y=np.sort(y), mode='markers', name='Experimental Data', marker=dict(color='gray', opacity=0.4)))
fig_var.add_trace(go.Scatter(y=y_rf_all[sorted_idx], mode='lines', name='AI Prediction', line=dict(color='#ff9900', width=3)))
fig_var.add_trace(go.Scatter(y=y_lr_all[sorted_idx], mode='lines', name='Linear Baseline', line=dict(color='cyan', dash='dash')))

fig_var.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
st.plotly_chart(fig_var, use_container_width=True)

# --- 7. FIXED FOOTER ---
st.markdown(f"""
    <div class="footer">
        Developed and Managed by <b>Mohammed Faheem M S</b> | Inconel 718 Machining Research 2026
    </div>
    """, unsafe_allow_html=True)

st.info(f"Model Variance identified at **{variance:.2f}%**. This justifies the requirement for an AI-driven predictive model.")
