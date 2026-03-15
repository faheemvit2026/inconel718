import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import math

# --- 1. DATASET & ML MODEL ---
# (Using your provided Inconel 718 data points)
data = {
    'Speed': [30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 92.9, 80, 70, 85, 80, 100, 60, 75, 75, 60, 60, 60, 60, 60, 60, 60],
    'Feed': [0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.12, 0.12, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    'DOC': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.1, 1, 1, 1, 1, 1, 1, 1],
    'Cooling': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], 
    'Temp': [580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 580, 560, 600, 630, 700, 750, 650, 620, 195, 710, 640, 750, 670, 720, 650, 760]
}
df = pd.DataFrame(data)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df[['Speed', 'Feed', 'DOC', 'Cooling']], df['Temp'])

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Inconel 718 Machining Suite", layout="wide")

st.markdown(f"""
    <style>
    .watermark {{
        position: fixed; bottom: 15px; right: 25px; opacity: 0.2;
        font-size: 22px; color: #555; z-index: 1000; pointer-events: none;
        font-family: sans-serif; font-weight: bold;
    }}
    .stMetric {{ background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #333; }}
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718 Machining Intelligence")
st.caption("Thermal Prediction & RPM Calculation Suite | Developed by mdfaheem")
st.divider()

# --- 3. SIDEBAR / INPUTS ---
st.sidebar.header("🕹️ CONTROL PANEL")

# RPM Calculation Section
st.sidebar.subheader("📐 RPM Calculator")
dia = st.sidebar.number_input("Rod Diameter (D in mm)", min_value=1.0, value=25.0, step=1.0)
in_speed = st.sidebar.slider("Cutting Speed (Vc in m/min)", 30, 200, 60)

# Calculate RPM using formula from image: (1000 * Vc) / (pi * D)
calculated_rpm = (1000 * in_speed) / (math.pi * dia)

st.sidebar.divider()

# Other AI Parameters
st.sidebar.subheader("🧠 AI Parameters")
in_feed = st.sidebar.slider("Feed Rate (f in mm/rev)", 0.05, 0.25, 0.10, step=0.01)
in_doc = st.sidebar.slider("Depth of Cut (ap in mm)", 0.1, 1.5, 0.5, step=0.1)
in_mode = st.sidebar.radio("Cooling Strategy", ["Dry", "MQL"])
in_cooling = 1 if in_mode == "MQL" else 0

# --- 4. CALCULATION RESULTS ---
# Predict Temp
prediction = model.predict([[in_speed, in_feed, in_doc, in_cooling]])[0]

top_col1, top_col2 = st.columns(2)

with top_col1:
    st.subheader("⚙️ Calculated Machine RPM")
    st.metric(label="Spindle Speed", value=f"{int(calculated_rpm)} RPM")
    st.caption(f"Formula: (1000 × {in_speed}) / (π × {dia})")

with top_col2:
    st.subheader("🔢 AI Predicted Temperature")
    st.metric(label="Tool-Chip Temp", value=f"{prediction:.1f} °C")

st.divider()

# --- 5. ANALOG GAUGE ---
col_gauge, col_info = st.columns([1.5, 1])

with col_gauge:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        title = {'text': "Live Thermal Monitor (°C)"},
        gauge = {
            'axis': {'range': [100, 1200]},
            'bar': {'color': "#00ffcc"},
            'steps': [
                {'range': [100, 600], 'color': "#1a472a"},
                {'range': [600, 950], 'color': "#47411a"},
                {'range': [950, 1200], 'color': "#471a1a"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 1000}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.write("### Machining Summary")
    st.success(f"**Target Material:** Inconel 718")
    st.info(f"**Rod Diameter:** {dia} mm")
    st.info(f"**Cooling Mode:** {in_mode}")
    
    if prediction > 950:
        st.error("⚠️ CRITICAL HEAT: High risk of tool plastic deformation.")
    elif prediction > 650:
        st.warning("⚠️ WARNING: Elevated temperatures detected.")
    else:
        st.success("✅ STABLE: Thermal load is within safe operational limits.")

    st.write("---")
    st.caption("Developer Signature: mdfaheem")
