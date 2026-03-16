import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.ensemble import RandomForestRegressor

# --- 1. CLEANED DATASET (Consistent Values Only) ---
clean_data = {
    'Speed': [30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 50, 70, 85, 90, 92.9, 80, 70, 85, 60],
    'Feed': [0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.06, 0.12, 0.12, 0.1, 0.1, 0.1, 0.08, 0.08, 0.15],
    'DOC': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.1],
    'Cooling': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
    'Temp': [580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 610, 820, 910, 980, 580, 560, 600, 630, 195]
}

df = pd.DataFrame(clean_data)
# Training on cleaned data: Speed, Feed, Depth of Cut, and Cooling Mode
X = df[['Speed', 'Feed', 'DOC', 'Cooling']]
y = df['Temp']
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# --- 2. LAYOUT & BRANDING ---
st.set_page_config(page_title="Inconel AI Command", layout="wide")

# Persistent Watermark
st.markdown("""
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 15px;
        color: rgba(150, 150, 150, 0.4);
        font-family: 'Courier New', monospace;
        font-weight: bold;
        z-index: 1000;
        pointer-events: none;
    }
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718 Machining Intelligence")
st.caption("AI Thermal Predictor & RPM Calculator | Developed by mdfaheem")
st.divider()

# --- 3. PARAMETER INPUTS (Manual Entry) ---
st.sidebar.header("🕹️ Parameters")

# Text/Number inputs for accuracy
dia = st.sidebar.number_input("Rod Diameter (mm)", value=25.0, format="%.2f")
in_speed = st.sidebar.number_input("Cutting Speed Vc (m/min)", value=60.0, format="%.2f")
in_feed = st.sidebar.number_input("Feed Rate f (mm/rev)", value=0.10, format="%.3f")
in_doc = st.sidebar.number_input("Depth of Cut ap (mm)", value=0.50, format="%.2f")
in_mode = st.sidebar.selectbox("Cooling Strategy", ["Dry", "MQL"])
in_cooling = 1 if in_mode == "MQL" else 0

# --- 4. ENGINE CALCULATIONS ---
# Spindle Speed Logic
calc_rpm = (1000 * in_speed) / (math.pi * dia)

# Machine Learning Prediction
# Prepare input array for the model
input_features = [[in_speed, in_feed, in_doc, in_cooling]]
prediction = model.predict(input_features)[0]

# --- 5. DASHBOARD VISUALS ---
# Top Row: Digital Readouts
col1, col2 = st.columns(2)
with col1:
    st.metric("Spindle Speed", f"{int(calc_rpm)} RPM")
    st.caption(f"Calculated for D={dia}mm")
with col2:
    st.metric("Predicted Temperature", f"{prediction:.1f} °C")
    st.caption("AI Prediction (Random Forest)")

st.divider()

# Bottom Row
