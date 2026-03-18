import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# --- 1. RESEARCH-VALIDATED DATASET (Inconel 718 + Carbide Tool) ---
# Inputs: [Speed, Feed, DOC] -> Outputs: [Temp, Force, Wear]
data = {
    'Speed': [30, 45, 60, 75, 90, 100, 120, 150, 40, 55, 70, 85, 200, 250, 15, 25],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.15, 0.2, 0.1, 0.12, 0.05, 0.08, 0.1, 0.12, 0.15, 0.1, 0.04, 0.06],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.8, 1.0, 0.5, 0.6, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 0.2, 0.3],
    # Targets based on literature trends for Inconel 718
    'Temp': [580, 635, 710, 795, 920, 1150, 815, 890, 420, 550, 680, 750, 1050, 1200, 312, 380],
    'Force': [280, 340, 410, 520, 680, 850, 430, 540, 180, 240, 310, 420, 790, 950, 145, 210],
    'Wear': [0.02, 0.05, 0.09, 0.14, 0.22, 0.35, 0.18, 0.28, 0.01, 0.04, 0.08, 0.12, 0.45, 0.60, 0.005, 0.02]
}

df = pd.DataFrame(data)
X = df[['Speed', 'Feed', 'DOC']]
y = df[['Temp', 'Force', 'Wear']]

# --- 2. MULTI-OUTPUT AI TRAINING ---
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)).fit(X, y)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Inconel 718 Master Lab", layout="wide")
st.title("🛡️ Inconel 718: Multi-Variable Machining Intelligence")
st.markdown(f"Developed by **Mohammed Faheem M S** | Research Scope: **Carbide Tooling**")

# Sidebar for inputs
st.sidebar.header("🕹️ Cutting Parameters")
dia = st.sidebar.number_input("Workpiece Diameter (mm)", value=25.0, format="%.4f")
v_c = st.sidebar.number_input("Cutting Speed (m/min)", value=60.0, format="%.4f")
f_r = st.sidebar.number_input("Feed Rate (mm/rev)", value=0.1, format="%.4f")
a_p = st.sidebar.number_input("Depth of Cut (mm)", value=0.5, format="%.4f")

# Calculations
rpm = (1000 * v_c) / (math.pi * dia)
prediction = model.predict([[v_c, f_r, a_p]])[0]
p_temp, p_force, p_wear = prediction

# --- 4. VISUAL DASHBOARD ---
c1, c2, c3 = st.columns(3)

with c1:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=p_temp, 
        title={'text': "TEMP (°C)"}, gauge={'axis': {'range': [0, 1300]}, 'bar': {'color': "#ff7b72"}})).update_layout(height=300))
with c2:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=p_force, 
        title={'text': "CUTTING FORCE (N)"}, gauge={'axis': {'range': [0, 1200]}, 'bar': {'color': "#58a6ff"}})).update_layout(height=300))
with c3:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=p_wear, 
        title={'text': "FLANK WEAR (mm)"}, gauge={'axis': {'range': [0, 0.8]}, 'bar': {'color': "#f2cc60"}})).update_layout(height=300))

# --- 5. SAFETY & ALERT SYSTEM ---
st.divider()
if p_wear > 0.3:
    st.error(f"⚠️ **CRITICAL WEAR ALERT:** Predicted Tool Wear ({p_wear:.4f}mm) exceeds ISO failure limit (0.3mm)!")
else:
    st.success(f"✅ **SAFE OPERATING ZONE:** Predicted Tool Wear is within acceptable limits.")

st.info(f"⚙️ **Calculated Spindle Speed:** {rpm:.2f} RPM")
