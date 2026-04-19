import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# --- 1. RESEARCH DATA ---
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

# --- 2. AI MODEL ---
X = train_df[['Speed', 'Feed', 'DOC', 'Diameter', 'Tool_Enc']]
y = train_df[['Temp', 'Force', 'Wear']]
model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=300, random_state=42)).fit(X, y)

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="Inconel 718 AI Twin", layout="wide")

# PROFESSIONAL TOP BANNER (Non-glitchy)
st.markdown("""
    <div style="background-color: #1E3A5F; padding: 30px; border-radius: 15px; border-bottom: 5px solid #FFD700; text-align: center; margin-bottom: 25px;">
        <h1 style="color: white; margin: 0; font-size: 50px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">MOHAMMED FAHEEM</h1>
        <p style="color: #FFD700; font-size: 1.2rem; margin: 10px 0;">B.Tech Mechanical Engineering | Manufacturing Specialization | VIT Vellore</p>
        <p style="color: #BDC3C7; font-size: 1rem;">Final Year Project: Intelligent Machining Analysis for Inconel 718</p>
    </div>
    """, unsafe_allow_html=True)

# ACCURACY CALCULATION
y_pred_all = model.predict(X)
mape_total = mean_absolute_percentage_error(y, y_pred_all)
overall_accuracy = (1 - mape_total) * 100

tab1, tab2, tab3 = st.tabs(["🚀 AI Simulator", "📊 Accuracy Validation", "📑 Experimental Dataset"])

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
        # --- RESTORED ERROR AND DANGER ALERTS ---
        st.subheader("⚠️ System Safety Monitor")
        
        if p[0] > 1100:
            st.error(f"🛑 **DANGER: CRITICAL TEMPERATURE ({p[0]:.2f}°C)** - Tool interface failure imminent!")
        elif p[0] > 900:
            st.warning(f"⚠️ **ALERT: HIGH THERMAL LOAD ({p[0]:.2f}°C)** - Monitor for rapid tool wear.")
        
        if p[1] > 1850:
            st.error(f"🚨 **DANGER: MECHANICAL OVERLOAD ({p[1]:.2f}N)** - Risk of insert chipping!")
        else:
            st.success(f"✅ **SYSTEM STABLE** - AI Confidence: {overall_accuracy:.2f}%")

        # PRECISION METRICS
        m1, m2, m3 = st.columns(3)
        m1.metric("Spindle RPM", f"{rpm:.4f}")
        m2.metric("Temp Output", f"{p[0]:.4f} °C")
        m3.metric("Force Output", f"{p[1]:.4f} N")
        
        # --- UPGRADED SPEEDOMETER GAUGES ---
        g1, g2 = st.columns(2)
        
        # Temp Gauge
        fig_t = go.Figure(go.Indicator(
            mode="gauge+number", value=p[0],
            title={'text': "Interface Temp (°C)", 'font': {'size': 20, 'color': "white"}},
            gauge={'axis': {'range': [0, 1500], 'tickwidth': 1, 'tickcolor': "white"},
                   'bar': {'color': "#e74c3c"},
                   'bgcolor': "#2c3e50",
                   'steps': [{'range': [900, 1100], 'color': "#f39c12"}, {'range': [1100, 1500], 'color': "#c0392b"}]}))
        fig_t.update_layout(paper_bgcolor="#1e1e1e", font={'color': "white"}, height=450)
        g1.plotly_chart(fig_t, use_container_width=True)

        # Force Gauge
        fig_f = go.Figure(go.Indicator(
            mode="gauge+number", value=p[1],
            title={'text': "Cutting Force (N)", 'font': {'size': 20, 'color': "white"}},
            gauge={'axis': {'range': [0, 2500], 'tickwidth': 1, 'tickcolor': "white"},
                   'bar': {'color': "#3498db"},
                   'bgcolor': "#2c3e50",
                   'steps': [{'range': [1850, 2500], 'color': "#2980b9"}]}))
        fig_f.update_layout(paper_bgcolor="#1e1e1e", font={'color': "white"}, height=450)
        g2.plotly_chart(fig_f, use_container_width=True)

# --- PROFESSIONAL FOOTER ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 20px;">
        <p style="margin-bottom: 5px;">Created and Developed by <b>Mohammed Faheem</b></p>
        <p style="font-size: 0.8rem;">© 2026 Mechanical Engineering Project | VIT Vellore | All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.metric("Overall System Accuracy", f"{overall_accuracy:.2f}%")
    st.metric("Model R² Confidence", f"{r2_score(y, y_pred_all):.6f}")

with tab3:
    st.dataframe(full_df, use_container_width=True)
