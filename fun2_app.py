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

# mdfaheem watermark
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

# --- 3. INPUT SECTION (Manual Entry) ---
st.sidebar.header("🕹️ Parameters")
dia = st.sidebar.number_input("Rod Diameter (mm)", value=25.0)
in_speed = st.sidebar.number_input("Cutting Speed Vc (m/min)", value=60.0)
in_feed = st.sidebar.number_input("Feed Rate f (mm/rev)", value=0.10, format="%.3f")
in_doc = st.sidebar.number_input("Depth of Cut ap (mm)", value=0.50)
in_mode = st.sidebar.selectbox("Cooling Strategy", ["Dry", "MQL"])
in_cooling = 1 if in_mode == "MQL" else 0

# --- 4. ENGINE CALCULATIONS ---
calc_rpm = (1000 * in_speed) / (math.pi * dia)
prediction = model.predict([[in_speed, in_feed, in_doc, in_cooling]])[0]

# --- 5. DASHBOARD VISUALS ---
# DIGITAL READOUTS
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Spindle Speed", f"{int(calc_rpm)} RPM")
with col2:
    st.metric("Predicted Temp", f"{prediction:.1f} °C")
with col3:
    status_label = "OPTIMAL" if prediction < 600 else "MODERATE" if prediction < 900 else "CRITICAL"
    st.metric("System Status", status_label)

st.divider()

# ANALOG GAUGE
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    title = {'text': "Analog Thermal Monitor (°C)", 'font': {'size': 20}},
    gauge = {
        'axis': {'range': [0, 1200]},
        'bar': {'color': "cyan"},
        'steps': [
            {'range': [0, 600], 'color': "#0e3321"},
            {'range': [600, 900], 'color': "#332c0e"},
            {'range': [900, 1200], 'color': "#330e0e"}]
    }
))
fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig, use_container_width=True)

# STATUS WARNINGS
st.subheader("📋 Operational Advisory")
if prediction >= 900:
    st.error(f"🔴 **CRITICAL TEMP:** Predicted at {prediction:.1f}°C. Extreme risk of tool failure. Reduce Vc or check MQL flow.")
elif prediction >= 600:
    st.warning(f"🟡 **MODERATE TEMP:** Predicted at {prediction:.1f}°C. Machining is stable but monitor tool wear closely.")
else:
    st.success(f"🟢 **OPTIMAL TEMP:** Predicted at {prediction:.1f}°C. Ideal thermal window for Inconel 718 longevity.")

st.info(f"Summary: Machining {dia}mm rod at {in_speed}m/min using {in_mode} cooling.")
