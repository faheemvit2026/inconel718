import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.ensemble import RandomForestRegressor

# --- 1. CLEANED DATASET ---
clean_data = {
    'Speed': [30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 50, 70, 85, 90, 92.9, 80, 70, 85, 60],
    'Feed': [0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.06, 0.12, 0.12, 0.1, 0.1, 0.1, 0.08, 0.08, 0.15],
    'DOC': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.1],
    'Cooling': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
    'Temp': [580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 610, 820, 910, 980, 580, 560, 600, 630, 195]
}

df = pd.DataFrame(clean_data)
X = df[['Speed', 'Feed', 'DOC', 'Cooling']]
y = df['Temp']
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# --- 2. LAYOUT & BRANDING ---
st.set_page_config(page_title="Inconel AI Analytics", layout="wide")

# Persistent mdfaheem watermark
st.markdown("""
    <style>
    .watermark {
        position: fixed; bottom: 10px; right: 15px; color: rgba(150, 150, 150, 0.4);
        font-family: 'Courier New', monospace; font-weight: bold; z-index: 1000; pointer-events: none;
    }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718 Machining Intelligence")
st.caption("AI Thermal Analytics Dashboard | Developed by mdfaheem")
st.divider()

# --- 3. INPUT SECTION (Manual Entry Only) ---
st.sidebar.header("🕹️ Parameters")
dia = st.sidebar.number_input("Rod Diameter (mm)", value=25.0, format="%.2f")
in_speed = st.sidebar.number_input("Cutting Speed Vc (m/min)", value=60
