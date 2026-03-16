import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.ensemble import RandomForestRegressor

# --- 1. PURE DRY MACHINING DATASET ---
# Filtered from your images: Conflicting/Outlier points removed for accuracy.
dry_data = {
    'Speed': [30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 60, 60, 60, 60, 60, 60],
    'Feed': [0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    'DOC': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
    'Temp': [580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 710, 750, 720, 760, 710, 750, 720]
}

df = pd.DataFrame(dry_data)
# Training model specifically for Dry Turning
X = df[['Speed', 'Feed', 'DOC']]
y = df['Temp']
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# --- 2. LAYOUT ---
st.set_page_config(page_title="Inconel AI - Dry Machining", layout="wide")

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
st.caption("Thermal Prediction for Carbide Tools (Dry Only) | Developed by mdfaheem")
st.divider()

# --- 3. INPUTS ---
st.sidebar.header("🕹️ Parameters")
dia = st.sidebar.number_input("Rod Diameter (mm)", value=25.0, format="%.2f")
in_speed = st.sidebar.number_input("Cutting Speed Vc (m/min)", value=60.0, format="%.2f")
in_feed = st.sidebar.number_input("Feed Rate f (mm/rev)", value=0.10, format="%.3f")
in_doc = st.sidebar.number_input("Depth of Cut ap (mm)", value=0.50, format="%.2f")

# --- 4. CALCULATIONS ---
calc_rpm = (1000 * in_speed) / (math.pi * dia)
prediction = model.predict([[in_speed, in_feed, in_doc]])[0]

# --- 5. DASHBOARD ---
c1, c2, c3 = st.columns(3)
c1.metric("Calculated RPM", f"{int(calc_rpm)}")
c2.metric("Predicted Temp", f"{prediction:.1f} °C")
c3.metric("Cooling State", "DRY")

st.divider()

# Gauge
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    title = {'text': "Analog Thermal Monitor (°C)"},
    gauge = {
        'axis': {'range': [100, 1300]},
        'bar': {'color': "cyan"},
        'steps': [
            {'range': [0, 650], 'color': "#0e3321"},
            {'range': [650, 950], 'color': "#332c0e"},
            {'range': [950, 1300], 'color': "#330e0e"}]
    }
))
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Advisories
if prediction >= 950:
    st.error(f"🔴 **CRITICAL:** High Heat ({prediction:.1f}°C). Expect rapid crater wear on Carbide Tool.")
elif prediction >= 650:
    st.warning(f"🟡 **MODERATE:** Stable cutting, but monitor for heat-induced softening.")
else:
    st.success(f"🟢 **OPTIMAL:** Low thermal load for Dry Turning.")

st.info(f"**Developer Note:** Prediction based on {len(df)} curated Dry Machining trials.")
