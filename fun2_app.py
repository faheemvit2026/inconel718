import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# --- 1. ENHANCED DATA CORE (Including your 10 New Semi-Finishing Trials) ---
data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300, 
              35.0, 40.0, 45.0, 50.0, 55.0, 42.0, 38.0, 52.0, 48.0, 60.0], # Added 10 new speeds
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12, 
             0.06, 0.08, 0.10, 0.08, 0.12, 0.07, 0.11, 0.09, 0.06, 0.05], # Added 10 new feeds
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 
            0.20, 0.25, 0.30, 0.35, 0.25, 0.40, 0.20, 0.30, 0.45, 0.50], # Added 10 new DOCs
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850, 
             428.5520, 465.1290, 512.8840, 548.2210, 582.4460, 495.3370, 445.6720, 561.9050, 529.1180, 598.7730] # Added 10 new Temps
}

df = pd.DataFrame(data)
X_feats = df[['Speed', 'Feed', 'DOC']]
y_actual = df['Temp']

# --- 2. AI MODEL TRAINING ---
# We use Random Forest to handle the mix of semi-finish and heavy-cutting data
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_feats, y_actual)
y_pred = rf_model.predict(X_feats)

# Calculate precision metrics
r2 = r2_score(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
accuracy = 100 - mape

# --- 3. UI SETUP ---
st.set_page_config(page_title="Inconel 718 Enhanced Research", layout="wide")
st.title("🛡️ Inconel 718: Thermal Precision Interface")
st.markdown(f"Developed by **Mohammed Faheem M S** | Dataset: {len(df)} Trials")
st.divider()

# --- 4. SIDEBAR INPUTS ---
st.sidebar.header("🕹️ Parameters")
dia = st.sidebar.number_input("Diameter (mm)", value=25.0, format="%.4f")
v_c = st.sidebar.number_input("Speed (m/min)", value=45.0, format="%.4f")
f_r = st.sidebar.number_input("Feed (mm/rev)", value=0.1, format="%.4f")
a_p = st.sidebar.number_input("DOC (mm)", value=0.3, format="%.4f")

# Calculations
calc_rpm = (1000 * v_c) / (math.pi * dia)
prediction = rf_model.predict([[v_c, f_r, a_p]])[0]

# --- 5. VISUAL GAUGES ---
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=calc_rpm, 
        number={'valueformat': ".4f", 'font': {'color': 'white'}},
        title={'text': "SPINDLE RPM", 'font': {'color': '#58a6ff'}},
        gauge={'axis': {'range': [0, 4000]}, 'bar': {'color': "#58a6ff"}})).update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)"), use_container_width=True)
with c2:
    st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=prediction, 
        number={'valueformat': ".4f", 'font': {'color': 'white'}},
        title={'text': "PREDICTED TEMP (°C)", 'font': {'color': '#ff7b72'}},
        gauge={'axis': {'range': [0, 1300]}, 'bar': {'color': "#ff7b72"}})).update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)"), use_container_width=True)

# --- 6. METRICS (4-Decimal Precision) ---
st.subheader("📊 Statistical Reliability")
k1, k2, k3, k4 = st.columns(4)
k1.metric("R² Score", f"{r2:.4f}")
k2.metric("Mean Squared Error", f"{mse:.4f}")
k3.metric("MAPE Error", f"{mape:.4f}%")
k4.metric("Model Accuracy", f"{accuracy:.4f}%")

st.divider()

# --- 7. REGRESSION VALIDATION ---
st.subheader("📈 Regression Parity: Actual vs. Predicted")
fig = go.Figure()
fig.add_trace(go.Scatter(x=[y_actual.min(), y_actual.max()], y=[y_actual.min(), y_actual.max()], mode='lines', line=dict(color='#8b949e', dash='dash'), name='Ideal Fit'))
fig.add_trace(go.Scatter(x=y_actual, y=y_pred, mode='markers', marker=dict(color='#ff7b72', size=8), name='Research Data'))
fig.update_layout(template="plotly_dark", height=500, xaxis_title="Actual Temp (°C)", yaxis_title="AI Predicted Temp (°C)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)
