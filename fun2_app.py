import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# --- 1. RAW EXPERIMENTAL DATA (DRY TURNING - INCONEL 718) ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}
df_ml = pd.DataFrame(dry_data)
features = ['Speed', 'Feed', 'DOC']
X, y = df_ml[features], df_ml['Temp']

# Advanced Regressors
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
lr_model = LinearRegression().fit(X, y)

# --- 2. PROFESSIONAL THEMING ---
st.set_page_config(page_title="Inconel 718 Research Hub | Mohammed Faheem M S", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9900; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: rgba(13, 17, 23, 0.95); color: #8b949e;
        text-align: center; padding: 12px; font-family: 'Segoe UI';
        border-top: 1px solid #30363d; z-index: 100;
    }
    .developer-tag { color: #ff9900; font-weight: 700; letter-spacing: 1px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BRANDED HEADER ---
c1, c2 = st.columns([4, 1.5])
with c1:
    st.title("🔬 Inconel 718: Thermal Predictive Intelligence")
    st.markdown("##### Experimental Analysis & Machine Learning Integration")
with c2:
    st.markdown(f"<div style='text-align: right; padding-top: 10px;'><span style='color: #8b949e;'>Project Lead:</span><br><span class='developer-tag'>Mohammed Faheem M S</span></div>", unsafe_allow_html=True)

st.divider()

# --- 4. DATA ACQUISITION (SIDEBAR) ---
st.sidebar.header("⚙️ System Inputs")
dia = st.sidebar.number_input("Workpiece Diameter (mm)", value=25.0, format="%.10f")
in_speed = st.sidebar.number_input("Cutting Speed Vc (m/min)", value=60.0, format="%.10f")
in_feed = st.sidebar.number_input("Feed Rate f (mm/rev)", value=0.1, format="%.10f")
in_doc = st.sidebar.number_input("Depth of Cut ap (mm)", value=0.5, format="%.10f")

# --- 5. HIGH-PRECISION ENGINE (NO ROUNDING) ---
calc_rpm = (1000 * in_speed) / (math.pi * dia)
rf_pred = rf_model.predict([[in_speed, in_feed, in_doc]])[0]
lr_pred = lr_model.predict([[in_speed, in_feed, in_doc]])[0]
variance_val = abs(rf_pred - lr_pred)
variance_pct = (variance_val / rf_pred) * 100

# --- 6. ANALOGUE INSTRUMENTATION ---
col_left, col_right = st.columns(2)

with col_left:
    fig_rpm = go.Figure(go.Indicator(
        mode = "gauge+number", value = calc_rpm,
        number = {'valueformat': ".6f", 'font': {'size': 35}},
        title = {'text': "SPINDLE FREQUENCY (RPM)", 'font': {'color': '#58a6ff'}},
        gauge = {'axis': {'range': [None, 4000], 'tickwidth': 1}, 'bar': {'color': "#58a6ff"},
                 'steps': [{'range': [0, 2500], 'color': '#161b22'}, {'range': [2500, 4000], 'color': '#0d1117'}]}
    ))
    fig_rpm.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig_rpm, use_container_width=True)

with col_right:
    fig_temp = go.Figure(go.Indicator(
        mode = "gauge+number", value = rf_pred,
        number = {'valueformat': ".6f", 'font': {'size': 35}},
        title = {'text': "PREDICTED INTERFACE TEMP (°C)", 'font': {'color': '#ff9900'}},
        gauge = {'axis': {'range': [None, 1300], 'tickwidth': 1}, 'bar': {'color': "#ff9900"},
                 'steps': [{'range': [0, 800], 'color': '#161b22'}, {'range': [800, 1300], 'color': '#0d1117'}],
                 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1100}}
    ))
    fig_temp.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig_temp, use_container_width=True)

# --- 7. ANALYTICAL METRICS ---
m1, m2, m3 = st.columns(3)
m1.metric("Linear Baseline (°C)", f"{lr_pred}")
m2.metric("AI Optimized (°C)", f"{rf_pred}")
m3.metric("Model Variance", f"{variance_pct}%")

# --- 8. VARIANCE GAP VISUALIZATION ---
st.subheader("📈 Statistical Variance & Non-Linearity Analysis")
y_rf_all = rf_model.predict(X)
y_lr_all = lr_model.predict(X)
sorted_idx = np.argsort(y)

fig_var = go.Figure()
fig_var.add_trace(go.Scatter(y=np.sort(y), mode='markers', name='Experimental Data', marker=dict(color='#8b949e', size=8, opacity=0.5)))
fig_var.add_trace(go.Scatter(y=y_rf_all[sorted_idx], mode='lines', name='RF Regressor (AI)', line=dict(color='#ff9900', width=3)))
fig_var.add_trace(go.Scatter(y=y_lr_all[sorted_idx], mode='lines', name='Linear Baseline', line=dict(color='#58a6ff', dash='dash')))

fig_var.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
st.plotly_chart(fig_var, use_container_width=True)

# --- 9. PERMANENT PROFESSIONAL FOOTER ---
st.markdown(f"""
    <div class="footer">
        © 2026 | Research Framework Developed by <span class="developer-tag">Mohammed Faheem M S</span> | 
        Mechanical Engineering - Inconel 718 Machining Studies
    </div>
    """, unsafe_allow_html=True)
