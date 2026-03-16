import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.ensemble import RandomForestRegressor

# --- 1. CLEANED DATASET ---
# I've kept only the most consistent values to ensure the AI logic is "proper"
clean_data = {
    'Speed': [30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 50, 70, 85, 90, 92.9, 80, 70, 85, 60],
    'Feed': [0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.06, 0.12, 0.12, 0.1, 0.1, 0.1, 0.08, 0.08, 0.15],
    'DOC': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.1],
    'Cooling': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
    'Temp': [580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 610, 820, 910, 980, 580, 560, 600, 630, 195]
}

df = pd.DataFrame(clean_data)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df[['Speed', 'Feed', 'DOC', 'Cooling']], df['Temp'])

# --- 2. LAYOUT & WATERMARK ---
st.set_page_config(page_title="Inconel AI Command", layout="wide")

# Simplified Watermark to ensure visibility
st.markdown("""
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: rgba(150, 150, 150, 0.5);
        font-family: monospace;
        z-index: 99;
    }
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718 Machining Intelligence")
st.write("Professional Thermal Prediction & RPM Calculator")

# --- 3. INPUT SECTION (Manual Typing) ---
st.sidebar.header("🕹️ Parameters")

# Manual inputs instead of sliders for precision
dia = st.sidebar.number_input("Rod Diameter (mm)", value=25.0)
in_speed = st.sidebar.number_input("Cutting Speed Vc (m/min)", value=60.0)
in_feed = st.sidebar.number_input("Feed Rate f (mm/rev)", value=0.10, format="%.3f")
in_doc = st.sidebar.number_input("Depth of Cut ap (mm)", value=0.50)
in_mode = st.sidebar.selectbox("Cooling Strategy", ["Dry", "MQL"])
in_cooling = 1 if in_mode == "MQL" else 0

# --- 4. CALCULATIONS ---
# RPM Calculation
calc_rpm = (1000 * in_speed) / (math.pi * dia)

# Temp Prediction
prediction = model.predict([[in_speed, in_feed, in_doc, in_cooling]])[0]

# --- 5. VISUAL DASHBOARD ---
st.divider()

# Top Row: Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Spindle Speed", f"{int(calc_rpm)} RPM")
col2.metric("Predicted Temp", f"{prediction:.1f} °C")
col3.metric("Cooling Mode", in_mode)

st.divider()

# Bottom Row: Analog Gauge
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    title = {'text': "Analog Heat Monitor (°C)"},
    gauge = {
        'axis': {'range': [None, 1200]},
        'bar': {'color': "cyan"},
        'steps': [
            {'range': [0, 600], 'color': "green"},
            {'range': [600, 950], 'color': "orange"},
            {'range': [950, 1200], 'color': "red"}]
    }
))

fig.update_layout(height=450)
st.plotly_
