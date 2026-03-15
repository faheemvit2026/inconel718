import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 1. DATASET PREPARATION ---
# Data extracted from your images (Dry and MQL only)
data = {
    'Speed': [30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 92.9, 80, 70, 85, 80, 100, 60, 75, 75, 60, 60, 60, 60, 60, 60, 60],
    'Feed': [0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.12, 0.12, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    'DOC': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.1, 1, 1, 1, 1, 1, 1, 1],
    'Cooling': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], # 0: Dry, 1: MQL
    'Temp': [580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 580, 560, 600, 630, 700, 750, 650, 620, 195, 710, 640, 750, 670, 720, 650, 760]
}

df = pd.DataFrame(data)

# Train the ML Model
X = df[['Speed', 'Feed', 'DOC', 'Cooling']]
y = df['Temp']
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Inconel 718 AI Analytics", layout="wide")

st.markdown(f"""
    <style>
    .watermark {{
        position: fixed; bottom: 15px; right: 25px; opacity: 0.2;
        font-size: 22px; color: #555; z-index: 1000; pointer-events: none;
        font-family: sans-serif; font-weight: bold;
    }}
    .stMetric {{ background-color: #1e2130; padding: 20px; border-radius: 12px; border: 1px solid #333; }}
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718 Machining Intelligence")
st.caption("Machine Learning Thermal Predictor | Developed by mdfaheem")
st.divider()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ PARAMETER INPUT")
in_speed = st.sidebar.slider("Speed (Vc)", 30, 200, 60)
in_feed = st.sidebar.slider("Feed (f)", 0.05, 0.20, 0.10, step=0.01)
in_doc = st.sidebar.slider("Depth of Cut (ap)", 0.1, 1.0, 0.5, step=0.1)
in_mode = st.sidebar.radio("Cooling Strategy", ["Dry", "MQL"])
in_cooling = 1 if in_mode == "MQL" else 0

# Predict using the trained model
prediction = model.predict([[in_speed, in_feed, in_doc, in_cooling]])[0]

# --- 4. DISPLAY LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    # Analog Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [100, 1200], 'tickwidth': 1},
            'bar': {'color': "#00ffcc"},
            'steps': [
                {'range': [100, 600], 'color': "#1a472a"},
                {'range': [600, 900], 'color': "#47411a"},
                {'range': [900, 1200], 'color': "#471a1a"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 1000}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔢 Digital System Readout")
    st.metric(label="Predicted Temperature", value=f"{prediction:.1f} °C")
    
    if prediction > 900:
        st.error("ALERT: CRITICAL THERMAL LOAD")
    elif prediction > 600:
        st.warning("STATUS: MODERATE HEAT")
    else:
        st.success("STATUS: OPTIMAL")

    st.write("### Prediction Context")
    st.info(f"Model trained on {len(df)} experimental data points from Inconel 718 trials.")
    st.write(f"**Cooling Mode:** {in_mode}")
    st.write(f"**Watermark:** mdfaheem")

st.divider()
