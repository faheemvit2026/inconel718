import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# --- 1. THE MULTI-VARIABLE DATASET ---
# Speed, Feed, DOC -> [Temp, Force, Wear]
data = {
    'Speed': [30, 60, 90, 120, 150, 45, 75, 100, 35, 55],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.15, 0.08, 0.1, 0.12, 0.06, 0.09],
    'DOC': [0.2, 0.5, 0.5, 0.6, 0.8, 0.3, 0.5, 0.6, 0.25, 0.4],
    # Targets: [Temperature (C), Cutting Force (N), Tool Wear (mm)]
    'Temp': [420, 650, 720, 850, 1100, 490, 680, 750, 445, 580],
    'Force': [150, 450, 480, 620, 950, 280, 460, 610, 210, 390],
    'Wear': [0.02, 0.08, 0.15, 0.22, 0.45, 0.04, 0.12, 0.18, 0.03, 0.09]
}

df = pd.DataFrame(data)
X = df[['Speed', 'Feed', 'DOC']]
y = df[['Temp', 'Force', 'Wear']] # Multi-target array

# --- 2. MULTI-OUTPUT AI TRAINING ---
# We wrap the Random Forest in a MultiOutputRegressor
base_rf = RandomForestRegressor(n_estimators=100, random_state=42)
multi_model = MultiOutputRegressor(base_rf).fit(X, y)

# --- 3. STREAMLIT INTERFACE ---
st.title("🛡️ Inconel 718: Total Machining Intelligence")
st.sidebar.header("Input Parameters")
v_c = st.sidebar.number_input("Speed (m/min)", value=60.0, format="%.2f")
f_r = st.sidebar.number_input("Feed (mm/rev)", value=0.1, format="%.2f")
a_p = st.sidebar.number_input("DOC (mm)", value=0.5, format="%.2f")

# Prediction logic
preds = multi_model.predict([[v_c, f_r, a_p]])[0]
temp_pred, force_pred, wear_pred = preds

# --- 4. TRIPLE ANALOGUE GAUGES ---
col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=temp_pred, 
        title={'text': "TEMP (°C)"}, gauge={'axis': {'range': [0, 1200]}, 'bar': {'color': "#ff7b72"}})).update_layout(height=300))
with col2:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=force_pred, 
        title={'text': "FORCE (N)"}, gauge={'axis': {'range': [0, 1500]}, 'bar': {'color': "#58a6ff"}})).update_layout(height=300))
with col3:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=wear_pred, 
        title={'text': "WEAR (mm)"}, gauge={'axis': {'range': [0, 0.6]}, 'bar': {'color': "#f2cc60"}})).update_layout(height=300))
