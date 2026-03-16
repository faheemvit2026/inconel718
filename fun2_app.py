import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# --- 1. RESEARCH DATA CORE ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}
df_ml = pd.DataFrame(dry_data)
X, y = df_ml[['Speed', 'Feed', 'DOC']], df_ml['Temp']

# Train Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
lr_model = LinearRegression().fit(X, y)

# --- 2. THEMING & PAGE CONFIG ---
st.set_page_config(page_title="Inconel 718 Research | Mohammed Faheem M S", layout="wide")

st.markdown("""
    <style>
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #0e1117; color: white;
        text-align: center; padding: 10px; font-size: 14px;
        border-top: 1px solid #333; z-index: 100;
    }
    .highlight { color: #ff9900; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER ---
st.title("🛡️ Inconel 718: Thermal Command Center")
st.markdown("Developed by **Mohammed Faheem M S**")
st.divider()

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Parameters")
dia = st.sidebar.number_input("Diameter (mm)", value=25.0, format="%.10f")
in_speed = st.sidebar.number_input("Speed Vc (m/min)", value=60.0, format="%.10f")
in_feed = st.sidebar.number_input("Feed f (mm/rev)", value=0.1, format="%.10f")
in_doc = st.sidebar.number_input("DOC ap (mm)", value=0.5, format="%.10f")

# Calculations
calc_rpm = (1000 * in_speed) / (math.pi * dia)
rf_pred = rf_model.predict([[in_speed, in_feed, in_doc]])[0]
lr_pred = lr_model.predict([[in_speed, in_feed, in_doc]])[0]
variance_pct = (abs(rf_pred - lr_pred) / rf_pred) * 100

# --- 5. ANALOGUE INSTRUMENTATION ---
col1, col2 = st.columns(2)
with col1:
    fig_rpm = go.Figure(go.Indicator(
        mode = "gauge+number", value = calc_rpm,
        number = {'valueformat': "f", 'font': {'size': 32}},
        title = {'text': "SPINDLE RPM", 'font': {'color': 'cyan'}},
        gauge = {'axis': {'range': [0, 4000]}, 'bar': {'color': "cyan"}}
    ))
    st.plotly_chart(fig_rpm, use_container_width=True)

with col2:
    fig_temp = go.Figure(go.Indicator(
        mode = "gauge+number", value = rf_pred,
        number = {'valueformat': "f", 'font': {'size': 32}},
        title = {'text': "AI TEMPERATURE (°C)", 'font': {'color': '#ff9900'}},
        gauge = {'axis': {'range': [0, 1300]}, 'bar': {'color': "#ff9900"}}
    ))
    st.plotly_chart(fig_temp, use_container_width=True)

# --- 6. CORE METRICS ---
m1, m2, m3 = st.columns(3)
m1.metric("Linear Baseline", f"{lr_pred}")
m2.metric("AI Optimized Prediction", f"{rf_pred}")
m3.metric("Model Variance", f"{variance_pct}%")

st.divider()

# --- 7. RESEARCH ANALYSIS SUITE (TABS) ---
st.subheader("📊 Research Data Analysis")
tab1, tab2, tab3 = st.tabs(["Non-Linearity Graph", "Regression Parity", "Residual Analysis"])

with tab1:
    st.write("Visualizing the 12.94% Physics Gap between models.")
    y_rf_all = rf_model.predict(X)
    y_lr_all = lr_model.predict(X)
    sorted_idx = np.argsort(y)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=np.sort(y), mode='markers', name='Actual Data', marker=dict(color='gray', opacity=0.4)))
    fig1.add_trace(go.Scatter(y=y_rf_all[sorted_idx], mode='lines', name='AI Model', line=dict(color='#ff9900', width=3)))
    fig1.add_trace(go.Scatter(y=y_lr_all[sorted_idx], mode='lines', name='Linear Trend', line=dict(color='cyan', dash='dash')))
    fig1.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.write("Actual vs. Predicted Temperature (Accuracy Line).")
    max_v = max(y.max(), y_rf_all.max())
    min_v = min(y.min(), y_rf_all.min())
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', name='Ideal Line', line=dict(color='white', dash='dash')))
    fig2.add_trace(go.Scatter(x=y, y=y_rf_all, mode='markers', name='RF Predictions', marker=dict(color='#ff9900', opacity=0.7)))
    fig2.update_layout(template="plotly_dark", height=400, xaxis_title="Actual (°C)", yaxis_title="Predicted (°C)")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.write("Distribution of prediction errors (Residuals).")
    residuals = y - y_rf_all
    fig3 = go.Figure()
    fig3.add_shape(type="line", x0=y_rf_all.min(), y0=0, x1=y_rf_all.max(), y1=0, line=dict(color="white", dash="dash"))
    fig3.add_trace(go.Scatter(x=y_rf_all, y=residuals, mode='markers', marker=dict(color='#ff9900')))
    fig3.update_layout(template="plotly_dark", height=400, xaxis_title="Predicted Temp", yaxis_title="Error (°C)")
    st.plotly_chart(fig3, use_container_width=True)

# --- 8. PERMANENT FOOTER ---
st.markdown(f"""
    <div class="footer">
        Developed by <span class="highlight">Mohammed Faheem M S</span> | Inconel 718 Machining Research 2026
    </div>
    """, unsafe_allow_html=True)
