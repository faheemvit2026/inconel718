import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# --- 1. DATASET ---
data = {
    'Speed': [30, 60, 90, 120, 150, 45, 75, 100, 35, 55],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.15, 0.08, 0.1, 0.12, 0.06, 0.09],
    'DOC': [0.2, 0.5, 0.5, 0.6, 0.8, 0.3, 0.5, 0.6, 0.25, 0.4],
    'Temp': [420, 650, 720, 850, 1100, 490, 680, 750, 445, 580],
    'Force': [150, 450, 480, 620, 950, 280, 460, 610, 210, 390],
    'Wear': [0.02, 0.08, 0.15, 0.22, 0.45, 0.04, 0.12, 0.18, 0.03, 0.09]
}
df = pd.DataFrame(data)
X = df[['Speed', 'Feed', 'DOC']]
y = df[['Temp', 'Force', 'Wear']]

# --- 2. AI MODEL ---
base_rf = RandomForestRegressor(n_estimators=100, random_state=42)
multi_model = MultiOutputRegressor(base_rf).fit(X, y)

# --- 3. UI LAYOUT ---
st.set_page_config(layout="wide")
st.title("🛡️ Inconel 718 Multi-Sensor Digital Twin")

# SIDEBAR
st.sidebar.header("Machine Settings")
dia = st.sidebar.number_input("Workpiece Diameter (mm)", value=25.0, format="%.2f")
v_c = st.sidebar.number_input("Cutting Speed (Vc - m/min)", value=60.0, format="%.2f")
f_r = st.sidebar.number_input("Feed Rate (f - mm/rev)", value=0.1, format="%.2f")
a_p = st.sidebar.number_input("Depth of Cut (ap - mm)", value=0.5, format="%.2f")

# MATH: Calculate RPM based on Diameter
# As Diameter goes UP, RPM must go DOWN to maintain the same Vc
calc_rpm = (1000 * v_c) / (math.pi * dia)

# AI PREDICTION
preds = multi_model.predict([[v_c, f_r, a_p]])[0]
t_pred, f_pred, w_pred = preds

# --- 4. GAUGES ---
# Top Row: Machine Physics
st.subheader("⚙️ Machine Dynamics")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=calc_rpm, 
        title={'text': "Calculated RPM"}, gauge={'axis': {'range': [0, 5000]}, 'bar': {'color': "#58a6ff"}})).update_layout(height=300))
with c2:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=t_pred, 
        title={'text': "Predicted Temp (°C)"}, gauge={'axis': {'range': [0, 1200]}, 'bar': {'color': "#ff7b72"}})).update_layout(height=300))

# Bottom Row: Tool & Force
st.subheader("🔧 Tool & Force Status")
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=f_pred, 
        title={'text': "Cutting Force (N)"}, gauge={'axis': {'range': [0, 1500]}, 'bar': {'color': "#f2cc60"}})).update_layout(height=300))
with c4:
    # Color coding for wear
    wear_color = "red" if w_pred > 0.3 else "green"
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=w_pred, 
        title={'text': "Flank Wear (mm)"}, gauge={'axis': {'range': [0, 0.5]}, 'bar': {'color': wear_color}})).update_layout(height=300))
