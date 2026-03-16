import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.ensemble import RandomForestRegressor

# --- 1. CLEANED DRY MACHINING DATASET ---
# Outliers (275C) and extreme contradictions removed for smoother prediction
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}

df_ml = pd.DataFrame(dry_data)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df_ml[['Speed', 'Feed', 'DOC']], df_ml['Temp'])

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Inconel AI Command Center", layout="wide")

st.markdown("""
    <style>
    .watermark {
        position: fixed; bottom: 10px; right: 15px; color: rgba(150, 150, 150, 0.3);
        font-family: 'Courier New', monospace; font-weight: bold; z-index: 1000; pointer-events: none;
    }
    .stMetric { background-color: #1e2130; padding: 10px; border-radius: 8px; border: 1px solid #333; }
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718 Machining Command Center")
st.caption("Real-time AI Thermal Prediction & Spindle Monitoring | Developed by mdfaheem")
st.divider()

# --- 3. INPUT SECTION (Manual Sidebar) ---
st.sidebar.header("🕹️ MACHINE INPUTS")
dia = st.sidebar.number_input("Workpiece Diameter D (mm)", value=25.0, step=1.0)
in_speed = st.sidebar.number_input("Cutting Speed Vc (m/min)", value=60.0, step=1.0)
in_feed = st.sidebar.number_input("Feed Rate f (mm/rev)", value=0.100, format="%.3f", step=0.005)
in_doc = st.sidebar.number_input("Depth of Cut ap (mm)", value=0.50, format="%.2f", step=0.05)

# --- 4. ENGINE CALCULATIONS ---
# Calculate Spindle RPM
calc_rpm = (1000 * in_speed) / (math.pi * dia)

# Predict Tool Temperature
prediction = model.predict([[in_speed, in_feed, in_doc]])[0]

# --- 5. DUAL SPEEDOMETER DISPLAY ---
col_rpm, col_temp = st.columns(2)

with col_rpm:
    # RPM Speedometer
    fig_rpm = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = calc_rpm,
        title = {'text': "SPINDLE SPEED (RPM)", 'font': {'size': 20, 'color': 'cyan'}},
        gauge = {
            'axis': {'range': [0, 4000], 'tickwidth': 1},
            'bar': {'color': "cyan"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 2000], 'color': "#0e2a33"},
                {'range': [2000, 3500], 'color': "#1a3a45"},
                {'range': [3500, 4000], 'color': "#451a1a"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 3800}
        }
    ))
    fig_rpm.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_rpm, use_container_width=True)

with col_temp:
    # Temperature Speedometer
    fig_temp = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        title = {'text': "TOOL TEMPERATURE (°C)", 'font': {'size': 20, 'color': '#ff9900'}},
        gauge = {
            'axis': {'range': [0, 1300], 'tickwidth': 1},
            'bar': {'color': "#ff9900"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 650], 'color': "#0e3321"},     # Safe
                {'range': [650, 950], 'color': "#332c0e"},    # Warning
                {'range': [950, 1300], 'color': "#330e0e"}],  # Danger
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1000}
        }
    ))
    fig_temp.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_temp, use_container_width=True)

st.divider()

# --- 6. DIGITAL STATUS SUMMARY ---
status_col1, status_col2 = st.columns([1, 2])

with status_col1:
    st.subheader("📊 System Summary")
    st.write(f"**Target:** Inconel 718")
    st.write(f"**Cooling:** Dry Turning")
    st.write(f"**Diameter:** {dia} mm")
    st.write(f"**Calculated RPM:** {int(calc_rpm)}")

with status_col2:
    if prediction >= 950:
        st.error(f"🚨 **CRITICAL HEAT:** {prediction:.1f}°C. Immediate risk of Carbide tool softening. Reduce cutting speed ($V_c$).
