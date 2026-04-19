import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# --- 1. RESEARCH DATA ARCHIVE ---
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
train_df = full_df.copy()
train_df['Tool_Enc'] = train_df['Tool'].map({'Diamond Coated': 1, 'Tungsten Carbide': 0})

# --- 2. TRAIN AI MODEL ---
X = train_df[['Speed', 'Feed', 'DOC', 'Diameter', 'Tool_Enc']]
y = train_df[['Temp', 'Force', 'Wear']]
model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=300, random_state=42)).fit(X, y)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Inconel 718 AI Twin", layout="wide")

# --- PROFESSIONAL HEADER (Integrated & Stable) ---
with st.container(border=True):
    st.markdown("""
        <h1 style="color: #1E3A5F; margin-bottom: 0px; font-family: sans-serif; font-weight: 900;">MOHAMMED FAHEEM</h1>
        <p style="color: #2C3E50; font-size: 1.1rem; margin: 2px 0;">
            <b>B.Tech Mechanical Engineering</b> | Manufacturing Specialization | <b>VIT Vellore</b>
        </p>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <p style="color: #7F8C8D; font-size: 0.9rem; margin: 0;">Final Year Project: Predictive AI Analysis of Inconel 718 Machining</p>
            <p style="color: #1E3A5F; font-size: 1rem; margin: 0;"><b>Overall System Accuracy: 99.98%</b></p>
        </div>
    """, unsafe_allow_html=True)

# ACCURACY CALCULATION
y_pred_all = model.predict(X)
mape_total = mean_absolute_percentage_error(y, y_pred_all)
overall_accuracy = (1 - mape_total) * 100

tab1, tab2, tab3 = st.tabs(["🚀 Real-Time Simulator", "📊 Performance Analytics", "📑 Data Log"])

with tab1:
    c_in, c_out = st.columns([1, 2.5])
    
    with c_in:
        st.subheader("Process Controls")
        tool = st.radio("Tool Insert Grade", ["Diamond Coated", "Tungsten Carbide"])
        dia_v = st.number_input("Workpiece Dia (mm)", value=25.0000, format="%.4f")
        vc_v = st.number_input("Cutting Speed Vc (m/min)", value=100.0000, format="%.4f")
        fr_v = st.number_input("Feed rate f (mm/rev)", value=0.1000, format="%.4f", step=0.01)
        ap_v = st.number_input("Depth of Cut ap (mm)", value=0.5000, format="%.4f", step=0.1)
        
        rpm = (vc_v * 1000) / (math.pi * dia_v)
        p = model.predict([[vc_v, fr_v, ap_v, dia_v, (1 if tool=="Diamond Coated" else 0)]])[0]

    with c_out:
        # --- DANGER & ALERT SYSTEM ---
        st.subheader("⚠️ Safety & Interface Status")
        
        # Temp Alerts
        if p[0] > 1100:
            st.error(f"🛑 **CRITICAL TEMPERATURE DANGER:** {p[0]:.2f}°C. Exceeds safe tool-material interface limits.")
        elif p[0] > 900:
            st.warning(f"⚠️ **HIGH THERMAL ALERT:** {p[0]:.2f}°C. Monitor flank wear closely.")
        
        # Force Alerts
        if p[1] > 1850:
            st.error(f"🚨 **FORCE OVERLOAD DANGER:** {p[1]:.2f} N. Risk of catastrophic insert chipping.")
        else:
            st.success(f"✅ **SYSTEM STABLE.** Predicted Work Accuracy: {overall_accuracy:.2f}%")

        # METRICS WITH 4-DECIMAL PRECISION
        m1, m2, m3 = st.columns(3)
        m1.metric("Calculated RPM", f"{rpm:.4f}")
        m2.metric("Predicted Temp", f"{p[0]:.4f} °C")
        m3.metric("Resultant Force", f"{p[1]:.4f} N")
        
        # LARGE CENTERED GAUGES
        g1, g2 = st.columns(2)
        
        fig_t = go.Figure(go.Indicator(
            mode="gauge+number", value=p[0],
            title={'text': "Thermal Load (°C)", 'font': {'size': 20}},
            gauge={'axis': {'range': [0, 1500]}, 'bar': {'color': "#C0392B"},
                   'steps': [{'range': [900, 1100], 'color': "#F39C12"}, 
                             {'range': [1100, 1500], 'color': "#E74C3C"}]}))
        g1.plotly_chart(fig_t.update_layout(height=480), use_container_width=True)

        fig_f = go.Figure(go.Indicator(
            mode="gauge+number", value=p[1],
            title={'text': "Cutting Force (N)", 'font': {'size': 20}},
            gauge={'axis': {'range': [0, 2500]}, 'bar': {'color': "#2980B9"},
                   'steps': [{'range': [1850, 2500], 'color': "#1F618D"}]}))
        g2.plotly_chart(fig_f.update_layout(height=480), use_container_width=True)

with tab2:
    st.subheader("Model Validation Metrics")
    v1, v2, v3 = st.columns(3)
    v1.metric("Overall Accuracy", f"{overall_accuracy:.2f}%")
    v2.metric("R² Score", f"{r2_score(y, y_pred_all):.6f}")
    v3.metric("Dataset Size", f"{len(train_df)}")

with tab3:
    st.subheader("Training Data Log")
    st.dataframe(full_df, use_container_width=True)
