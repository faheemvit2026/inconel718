import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- 1. DATA SOURCE ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}
df = pd.DataFrame(dry_data)
X, y = df[['Speed', 'Feed', 'DOC']], df['Temp']

# --- 2. RESIDUAL CALCULATION ---
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
y_pred = rf.predict(X)
residuals = y - y_pred  # Actual - Predicted

# --- 3. PLOTLY RESIDUAL GRAPH ---
fig_res = go.Figure()

# Zero Error Line
fig_res.add_shape(type="line", x0=y_pred.min(), y0=0, x1=y_pred.max(), y1=0,
                  line=dict(color="white", width=2, dash="dash"))

# Residual Points
fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                             marker=dict(color='#ff9900', size=10, 
                                         line=dict(width=1, color='white'))))

fig_res.update_layout(
    title="Residual Plot: AI Prediction Errors",
    xaxis_title="Predicted Temperature (°C)",
    yaxis_title="Residual (Error in °C)",
    template="plotly_dark",
    height=450
)

st.plotly_chart(fig_res, use_container_width=True)
st.markdown("Developed by **Mohammed Faheem M S**")
