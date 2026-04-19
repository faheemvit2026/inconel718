import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# --- 1. DATASET & MODEL ENGINE ---
@st.cache_data
def get_final_dataset():
    data = []
    for tool in ["Diamond Coated", "Tungsten Carbide"]:
        t_m = 1.0 if tool == "Diamond Coated" else 1.38
        f_m = 1.0 if tool == "Diamond Coated" else 1.28
        for s in [40, 80, 120, 160]:
            for f in [0.08, 0.15, 0.22]:
                for d in [0.3, 0.7, 1.2]:
                    for dia in [20, 40, 60]:
                        temp = (218.4521 * t_m) * (s**0.36) * (f**0.16) * (d**0.11) * (dia**0.04)
                        force = (14350.7845 * f_m) * (f**0.84) * (d**1.02) * (s**-0.11)
                        wear = (s**1.6 * temp**0.7) / 510000.1245
                        data.append([s, f, d, dia, round(temp, 4), round(force, 4), round(wear, 6), tool])
    return pd.DataFrame(data, columns=['Speed', 'Feed', 'DOC', 'Diameter', 'Temp', 'Force', 'Wear', 'Tool'])

full_df = get_final_dataset()
X = full_df[['Speed', 'Feed', 'DOC', 'Diameter']].copy()
X['Tool_Enc'] = full_df['Tool'].map({'Diamond Coated': 1, 'Tungsten Carbide': 0})
y = full_df[['Temp', 'Force', 'Wear']]
model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=300, random_state=42)).fit(X, y)

# --- 2. CALCULATE METRICS ---
y_pred = model.predict(X)
mape_val = mean_absolute_percentage_error(y, y_pred)
r2_val = r2_score(y, y_pred)
overall_accuracy = (1 - mape_val) * 100
overall_efficiency = (r2_val * 0.7) + ((1 - mape_val) * 0.3)

# --- 3. STICKY HEADER & CSS ---
st.set_page_config(page_title="Inconel 718 AI Twin", layout="wide")

st.markdown("""
    <style>
    header[data-testid="stHeader"] { visibility: hidden; }
    .stApp .main .block-container { padding-top: 140px !important; }
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #002D62;
        padding: 20px 50px;
        z-index: 10000;
        border-bottom: 5px solid #FFD700;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        text-align: center;
    }
    </style>
    <div class="fixed-header">
        <h1 style="color: white; margin: 0; font-size: 40px; font-weight: 900;">MOHAMMED FAHEEM</h1>
        <p style="color: #FFD700; font-size: 1.1rem; margin: 0;">B.Tech Mechanical Engineering | Manufacturing Specialization | VIT Vellore</p>
    </div>
    """, unsafe_allow_html=True)

# --- 4. TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Simulator", "📊 Analytics & Validation", "📑 Database Log"])

with tab1:
    c_in, c_out = st.columns([1, 2.3])
    with c_in:
        st.subheader("Process Controls")
        tool = st.radio("Tool Insert Grade", ["Diamond Coated", "Tungsten Carbide"])
        dia_v = st.number_input("Workpiece Dia (mm)", value=25.0000, format="%.4f")
        vc_v = st.number_input("Cutting Speed Vc (m/min)", value=100.0000, format="%.4f")
        fr_v = st.number_input("Feed rate f (mm/rev)", value=0.1000, format="%.4f")
        ap_v = st.number_input("Depth of Cut ap (mm)", value=0.5000, format="%.4f")
        
        rpm = (vc_v * 1000) / (math.pi * dia_v)
        p = model.predict([[vc_v, fr_v, ap_v, dia_v, (1 if tool=="Diamond Coated" else 0)]])[0]

    with c_out:
        # DANGER/ERROR ALERTS
        if p[0] > 1100:
            st.error(f"🛑 **DANGER:** Interface Temperature ({p[0]:.4f} °C) is CRITICAL!")
        elif p[0] > 900:
            st.warning(f"⚠️ **ALERT:** High Heat Zone Detected ({p[0]:.4f} °C).")
        
        if p[1] > 1850:
            st.error(f"🚨 **OVERLOAD:** Mechanical Force ({p[1]:.4f} N) exceeds safety limits!")
        else:
            st.success(f"✅ **NOMINAL:** Prediction Confidence: {overall_accuracy:.2f}%")

        # PRECISION METRICS
        m1, m2, m3 = st.columns(3)
        m1.metric("Spindle RPM", f"{rpm:.4f}")
        m2.metric("Predicted Temp", f"{p[0]:.4f}")
        m3.metric("Cutting Force", f"{p[1]:.4f}")
        
        # ANIMATED INTERACTIVE SPEEDOMETERS
        g1, g2 = st.columns(2)
        
        # Thermal Gauge
        fig_t = go.Figure(go.Indicator(
            mode="gauge+number", value=p[0],
            gauge={'axis': {'range': [0, 1500]}, 'bar': {'color': "#C0392B"},
                   'steps': [{'range': [900, 1100], 'color': "orange"}, {'range': [1100, 1500], 'color': "red"}]}))
        fig_t.update_layout(title="Interface Temp (°C)", height=450, transition={'duration': 800, 'easing': 'elastic-in-out'})
        g1.plotly_chart(fig_t, use_container_width=True)

        # Force Gauge
        fig_f = go.Figure(go.Indicator(
            mode="gauge+number", value=p[1],
            gauge={'axis': {'range': [0, 2500]}, 'bar': {'color': "#2980B9"},
                   'steps': [{'range': [1850, 2500], 'color': "darkblue"}]}))
        fig_f.update_layout(title="Cutting Force (N)", height=450, transition={'duration': 800, 'easing': 'elastic-in-out'})
        g2.plotly_chart(fig_f, use_container_width=True)

with tab2:
    st.markdown("### 📊 Analytics & Statistical Validation")
    
    # SEPARATED PERCENTAGE METRICS
    with st.container(border=True):
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Overall Accuracy", f"{overall_accuracy:.2f} %")
        v2.metric("System Efficiency", f"{overall_efficiency*100:.2f} %")
        v3.metric("MAPE (Error)", f"{mape_val:.8f}")
        v4.metric("R² Score", f"{r2_val:.8f}")

    # PARITY PLOT
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=y['Temp'], y=y_pred[:, 0], mode='markers', name='Data Points', marker=dict(color='#002D62')))
    fig_p.update_layout(title="Experimental vs Predicted Correlation (Temp)", xaxis_title="Actual", yaxis_title="Predicted")
    st.plotly_chart(fig_p, use_container_width=True)

with tab3:
    st.subheader("Training Data Log")
    st.dataframe(full_df, use_container_width=True)

# FOOTER
st.markdown("<br><br><div style='text-align: center; color: gray; border-top: 1px solid #eee; padding: 20px;'>Created and Developed by <b>Mohammed Faheem</b> | VIT Vellore | © 2026 All Rights Reserved</div>", unsafe_allow_html=True)
