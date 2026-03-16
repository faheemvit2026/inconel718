import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.ensemble import RandomForestRegressor

# --- 1. NEW VERIFIED DATASET (DRY ONLY) ---
# Extracted directly from your provided Excel sheet
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 75, 50, 70, 90, 60, 80, 100, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.1, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 275, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 275, 620, 690, 750, 710, 780, 860, 600, 720, 850]
}

df_ml = pd.DataFrame(dry_data)
# Training model on Speed (Vc), Feed (f), and Depth of Cut (ap)
X = df_ml[['Speed', 'Feed', 'DOC']]
y = df_ml['Temp']
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Inconel 718 AI Analytics", layout="wide")

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

st.title("🛡️ Inconel 718: Dry Machining Intelligence")
st.caption("AI Thermal Predictor | Data Source: Verified Dry Experimental Results | Developed by mdfaheem")
st.divider()

# --- 3. INPUT SECTION (Manual Entry) ---
st.sidebar.header("🕹️ Parameters")
dia = st.sidebar.number_input("Rod Diameter (mm)", value=25.0, step=1.0)
in_speed = st.sidebar.number_input("Cutting Speed Vc (m/min)", value=60.0, step=1.0)
in_feed = st.sidebar.number_input("Feed Rate f (mm/rev)", value=0.10, format="%.3f", step=0.01)
in_doc = st.sidebar.number_input("Depth of Cut ap (mm)", value=0.50, format="%.2f", step=0.1)

# --- 4. ENGINE CALCULATIONS ---
# Spindle RPM Formula
calc_rpm = (1000 * in_speed) / (math.pi * dia)
# Machine Learning Prediction
prediction = model.predict([[in_speed, in_feed, in_doc]])[0]

# --- 5. DASHBOARD VISUALS ---
# DIGITAL READOUTS
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Spindle Speed", f"{int(calc_rpm)} RPM")
with col2:
    st.metric("Predicted Temp", f"{prediction:.1f} °C")
with col3:
    status_label = "OPTIMAL" if prediction < 650 else "MODERATE" if prediction < 950 else "CRITICAL"
    st.metric("System Status", status_label)

st.divider()

# ANALOG GAUGE
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    title = {'text': "Analog Thermal Monitor (°C)", 'font': {'size': 20}},
    gauge = {
        'axis': {'range': [0, 1300]},
        'bar': {'color': "cyan"},
        'steps': [
            {'range': [0, 650], 'color': "#0e3321"},     # Green Zone
            {'range': [650, 950], 'color': "#332c0e"},    # Orange Zone
            {'range': [950, 1300], 'color': "#330e0e"}]   # Red Zone
    }
))
fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig, use_container_width=True)

# OPERATIONAL ADVISORIES
st.subheader("📋 Operational Advisory")
if prediction >= 950:
    st.error(f"🔴 **CRITICAL TEMP:** Predicted at {prediction:.1f}°C. Extreme risk for Carbide tools in dry conditions. Reduce Speed or Feed immediately.")
elif prediction >= 650:
    st.warning(f"🟡 **MODERATE TEMP:** Predicted at {prediction:.1f}°C. Machining is possible but monitor tool tip wear and crater formation.")
else:
    st.success(f"🟢 **OPTIMAL TEMP:** Predicted at {prediction:.1f}°C. Safe thermal window for dry machining of Inconel 718.")

st.info(f"**Analytics Note:** Model trained exclusively on Dry Machining trials for {dia}mm rod diameter.")
