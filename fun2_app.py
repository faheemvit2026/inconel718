import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.ensemble import RandomForestRegressor

# --- 1. CLEANED DRY DATASET ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}

df_ml = pd.DataFrame(dry_data)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df_ml[['Speed', 'Feed', 'DOC']], df_ml['Temp'])

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Inconel AI Dashboard", layout="wide")

st.markdown("""
    <style>
    .watermark {
        position: fixed; bottom: 10px; right: 15px; color: rgba(150, 150, 150, 0.3);
        font-family: 'Courier New', monospace; font-weight: bold; z-index: 1000; pointer-events: none;
    }
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718 Machining Command Center")
st.caption("AI Thermal Analytics & RPM Monitoring (Dry Only) | Developed by mdfaheem")
st.divider()

# --- 3. INPUTS ---
st.sidebar.header("🕹️ MACHINE INPUTS")
dia = st.sidebar.number_input("Workpiece Diameter D (mm)", value=25.0, step=1.0)
in_speed = st.sidebar.number_input("Cutting Speed Vc (m/min)", value=60.0, step=1.0)
in_feed = st.sidebar.number_input("Feed Rate f (mm/rev)", value=0.100, format="%.3f", step=0.005)
in_doc = st.sidebar.number_input("Depth of Cut ap (mm)", value=0.50, format="%.2f", step=0.05)

# --- 4. ENGINE CALCULATIONS ---
calc_rpm = (1000 * in_speed) / (math.pi * dia)
prediction = model.predict([[in_speed, in_feed, in_doc]])[0]

# --- 5. DUAL SPEEDOMETER GAUGES ---
col_rpm, col_temp = st.columns(2)

with col_rpm:
    fig_rpm = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = calc_rpm,
        title = {'text': "SPINDLE SPEED (RPM)", 'font': {'size': 22, 'color': 'cyan'}},
        gauge = {
            'axis': {'range': [0, 4000]},
            'bar': {'color': "cyan"},
            'steps': [
                {'range': [0, 2000], 'color': "#0e2a33"},
                {'range': [2000, 3500], 'color': "#1a3a45"},
                {'range': [3500, 4000], 'color': "#451a1a"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 3800}
        }
    ))
    fig_rpm.update_layout(height=400, margin=dict(l=40, r=40, t=80, b=40))
    st.plotly_chart(fig_rpm, use_container_width=True)

with col_temp:
    fig_temp = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        title = {'text': "TOOL TEMPERATURE (°C)", 'font': {'size': 22, 'color': '#ff9900'}},
        gauge = {
            'axis': {'range': [0, 1300]},
            'bar': {'color': "#ff9900"},
            'steps': [
                {'range': [0, 650], 'color': "#0e3321"},
                {'range': [650, 950], 'color': "#332c0e"},
                {'range': [950, 1300], 'color': "#330e0e"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1000}
        }
    ))
    fig_temp.update_layout(height=400, margin=dict(l=40, r=40, t=80, b=40))
    st.plotly_chart(fig_temp, use_container_width=True)

st.divider()

# --- 6. ADVISORY SECTION ---
if prediction >= 950:
    st.error(f"🚨 **CRITICAL HEAT:** {prediction:.1f}°C. Immediate risk of Carbide tool softening. Reduce cutting speed ($V_c$).")
elif prediction >= 650:
    st.warning(f"⚠️ **MODERATE LOAD:** {prediction:.1f}°C. Stable machining, but monitor for rapid crater wear.")
else:
    st.success(f"✅ **OPTIMAL WINDOW:** {prediction:.1f}°C. Thermal parameters are safe for dry turning.")

st.info(f"Summary: Machining {dia}mm rod at {in_speed}m/min. Predicted Spindle Speed: {int(calc_rpm)} RPM.")
