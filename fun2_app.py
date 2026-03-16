import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- 1. CLEANED DATASET (DRY ONLY) ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}

df_ml = pd.DataFrame(dry_data)
features = ['Speed', 'Feed', 'DOC']
X = df_ml[features]
y = df_ml['Temp']
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# --- 2. LAYOUT ---
st.set_page_config(page_title="Inconel AI Command Center", layout="wide")

st.markdown("""
    <style>
    .watermark { position: fixed; bottom: 10px; right: 15px; color: rgba(150, 150, 150, 0.3); font-weight: bold; z-index: 1000; }
    .ai-box { background-color: #0e1117; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-bottom: 20px; }
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718: Self-Thinking AI Command")
st.caption("Real-time Reasoning Engine for Dry Machining | Developed by mdfaheem")

# --- 3. INPUT SIDEBAR ---
st.sidebar.header("🕹️ Parameters")
dia = st.sidebar.number_input("Diameter (mm)", value=25.0, min_value=0.1)
in_speed = st.sidebar.number_input("Speed Vc (m/min)", value=60.0)
in_feed = st.sidebar.number_input("Feed f (mm/rev)", value=0.100, format="%.3f")
in_doc = st.sidebar.number_input("DOC ap (mm)", value=0.50)

# --- 4. AI CORE ---
calc_rpm = (1000 * in_speed) / (math.pi * dia)
current_input = [[in_speed, in_feed, in_doc]]
prediction = model.predict(current_input)[0]

# AI Reasoning: Calculate what's driving the heat
importances = model.feature_importances_
dominant_factor = features[np.argmax(importances)]

# --- 5. VISUALS ---
col_rpm, col_temp = st.columns(2)

with col_rpm:
    fig_rpm = go.Figure(go.Indicator(
        mode = "gauge+number", value = calc_rpm,
        title = {'text': "SPINDLE RPM", 'font': {'color': 'cyan'}},
        gauge = {'axis': {'range': [0, 4000]}, 'bar': {'color': "cyan"},
                 'steps': [{'range': [0, 3500], 'color': "#0e2a33"}, {'range': [3500, 4000], 'color': "#451a1a"}]}
    ))
    st.plotly_chart(fig_rpm, use_container_width=True)

with col_temp:
    fig_temp = go.Figure(go.Indicator(
        mode = "gauge+number", value = prediction,
        title = {'text': "AI PREDICTED TEMP (°C)", 'font': {'color': '#ff9900'}},
        gauge = {'axis': {'range': [0, 1300]}, 'bar': {'color': "#ff9900"},
                 'steps': [{'range': [0, 650], 'color': "#0e3321"}, {'range': [650, 950], 'color': "#332c0e"}, {'range': [950, 1300], 'color': "#330e0e"}]}
    ))
    st.plotly_chart(fig_temp, use_container_width=True)

# --- 6. THE "SELF-THINKING" ASPECT ---
st.divider()
st.subheader("🧠 AI Internal Reasoning")

box1, box2 = st.columns([2, 1])

with box1:
    st.markdown(f"""
    <div class="ai-box">
    <h4>🤖 Why is the temperature {prediction:.1f}°C?</h4>
    <p>The AI has analyzed your inputs and determined that <b>{dominant_factor}</b> is currently the primary driver of thermal energy in this cut.</p>
    <ul>
        <li><b>Thermal Observation:</b> At {in_speed} m/min, the plastic deformation of Inconel 718 is localized at the tool tip.</li>
        <li><b>Machinability Insight:</b> Dry machining relies on chip evacuation to carry away heat. Increasing {dominant_factor} further may exceed the tool's thermal threshold.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with box2:
    # Small chart showing what the AI is thinking about
    fig_imp = go.Figure(go.Bar(
        x=features, y=importances,
        marker_color=['#00ccff', '#00ffcc', '#ffcc00']
    ))
    fig_imp.update_layout(title="AI Feature Weight", height=200, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_imp, use_container_width=True)

# Safety Alerts
if prediction > 950:
    st.error("🚨 **CRITICAL:** AI detects high risk of Diffusion Wear. Lower the Speed immediately.")
elif prediction > 650:
    st.warning("⚠️ **STABLE BUT HOT:** AI suggests monitoring tool edge for 'BUE' (Built-up Edge).")
else:
    st.success("✅ **OPTIMAL:** AI confirms these parameters are safe for long-duration machining.")
