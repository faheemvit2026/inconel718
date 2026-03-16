import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import math

# --- 1. CLEANED DATASET (Outliers and conflicting values removed) ---
# Filtered for consistency in Inconel 718 turning trials (Dry vs MQL)
clean_data = {
    'Speed': [30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 50, 75, 100, 55, 70, 85, 90, 92.9, 80, 70, 85, 60],
    'Feed': [0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.06, 0.06, 0.06, 0.12, 0.12, 0.12, 0.1, 0.1, 0.1, 0.08, 0.08, 0.15],
    'DOC': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.1],
    'Cooling': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], # 0: Dry, 1: MQL
    'Temp': [580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 610, 790, 940, 715, 820, 910, 980, 580, 560, 600, 630, 195]
}

df = pd.DataFrame(clean_data)
# Training a robust model on the cleaned data
X = df[['Speed', 'Feed', 'DOC', 'Cooling']]
y = df['Temp']
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="mdfaheem | Machining AI", layout="wide")

# Custom CSS for the Watermark and Dark Command Center Look
st.markdown(f"""
    <style>
    .watermark {{
        position: fixed; bottom: 20px; right: 30px; opacity: 0.15;
        font-size: 24px; color: #888; z-index: 1000; pointer-events: none;
        font-family: 'Courier New', monospace; font-weight: bold;
    }}
    .stNumberInput div div input {{ background-color: #1e2130; color: #00FFCC !important; }}
    .stMetric {{ background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }}
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718 Machining Intelligence")
st.caption("Cleaned Dataset Predictor & RPM Calculator | Developed by mdfaheem")
st.divider()
