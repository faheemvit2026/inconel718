import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# --- 1. DATA SOURCE ---
dry_data = {
    'Speed': [60, 75, 90, 80, 100, 60, 75, 60, 30, 40, 50, 60, 80, 100, 30, 45, 60, 75, 100, 50, 50, 50, 75, 75, 75, 100, 100, 40, 55, 70, 85, 100, 60, 60, 60, 90, 90, 90, 60, 75, 90, 80, 100, 60, 75, 60, 75, 90, 50, 70, 90, 60, 80, 100, 200, 300],
    'Feed': [0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.15, 0.05, 0.05, 0.08, 0.08, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.1, 0.06, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.12, 0.12],
    'DOC': [0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.8, 0.5, 0.8, 0.2, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6],
    'Temp': [650, 690, 720, 700, 750, 650, 620, 710, 580, 635, 710, 795, 920, 1150, 760, 815, 890, 960, 1050, 610, 685, 740, 790, 860, 765, 940, 880, 640, 715, 820, 910, 985, 755, 810, 865, 920, 980, 950, 650, 690, 720, 700, 750, 650, 620, 650, 690, 720, 620, 690, 750, 710, 780, 860, 720, 850]
}
df = pd.DataFrame(dry_data)
X, y = df[['Speed', 'Feed', 'DOC']], df['Temp']

# --- 2. MODEL TRAINING ---
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
lr = LinearRegression().fit(X, y)

y_rf = rf.predict(X)
y_lr = lr.predict(X)

# --- 3. PLOTLY REGRESSION GRAPH ---
fig = go.Figure()

# Perfect Prediction Line
max_val = max(y.max(), y_rf.max())
min_val = min(y.min(), y_rf.min())
fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                         mode='lines', name='Ideal Prediction (Identity)', 
                         line=dict(color='white', dash='dash', width=1)))

# Random Forest Predictions
fig.add_trace(go.Scatter(x=y, y=y_rf, mode='markers', name='AI Prediction (RF)', 
                         marker=dict(color='#ff9900', size=10, opacity=0.7, 
                                     line=dict(width=1, color='white'))))

# Linear Regression Predictions
fig.add_trace(go.Scatter(x=y, y=y_lr, mode='markers', name='Linear Baseline', 
                         marker=dict(color='cyan', size=8, symbol='x', opacity=0.5)))

fig.update_layout(
    title="Regression Analysis: Actual vs. Predicted Temperature",
    xaxis_title="Actual Experimental Temperature (°C)",
    yaxis_title="Model Predicted Temperature (°C)",
    template="plotly_dark",
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("Developed by **Mohammed Faheem M S**")
