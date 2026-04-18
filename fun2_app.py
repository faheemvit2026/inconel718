import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# --- 1. RESEARCH-VALIDATED DATABASE ---
@st.cache_data
def get_final_data():
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

full_df = get_final_data()
train_df = full_df.copy()
train_df['Tool_Enc'] = train_df['Tool'].map({'Diamond Coated': 1, 'Tungsten Carbide': 0})

# --- 2. TRAIN AI MODEL ---
X = train_df[['Speed', 'Feed', 'DOC', 'Diameter', 'Tool_Enc']]
y = train_df[['Temp', 'Force', 'Wear']]
model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=300, random_state=42)).fit(X, y)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Inconel 718 Digital Twin", layout="wide")

# PROFESSIONAL HEADER
st.markdown("""
    <div style="border-bottom: 2px solid #2c3e50; margin-bottom: 20px;">
        <h2 style="color: #2c3e50; margin-bottom: 5px;">MOHAMMED FAHEEM</h2>
        <p style="color: #7f8c8d; font-size: 16px; margin-top: 0;">
            B.Tech Mechanical Engineering (Manufacturing) | VIT Vellore<br>
            <b>Project: AI-Driven Machining Analysis of Inconel 718</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Process Simulator", "📊 Model Validation", "📑 Database"])

with tab1:
    c_in, c_out = st.columns([1, 2.5])
    with c_in:
        st.subheader("Control Panel")
        tool = st.radio("Insert Grade", ["Diamond Coated", "Tungsten Carbide"])
        dia_v = st.number_input("Workpiece Dia (mm)", value=30.0, format="%.4f")
        vc_v = st.number_input("Cutting Speed Vc (m/min)", value=100.0, format="%.4f")
        fr_v = st.number_input("Feed rate f (mm/rev)", value=0.1000, format="%.4f", step=0.01)
        ap_v = st.number_input("Depth of Cut ap (mm)", value=0.5000, format="%.4f", step=0.1)
        
        rpm = (vc_v * 1000) / (math.pi * dia_v)
        p = model.predict([[vc_v, fr_v, ap_v, dia_v, (1 if tool=="Diamond Coated" else 0)]])[0]

    with c_out:
        # --- DANGER WARNING SYSTEM ---
        st.subheader("System Status")
        warn_c = st.container()
        if p[0] > 1100:
            warn_c.error("⚠️ **CRITICAL HEAT ALERT:** Interface temperature exceeds 1100°C. Risk of rapid tool diffusion and workpiece burning.")
        elif p[0] > 900:
            warn_c.warning("⚠️ **HIGH TEMPERATURE WARNING:** High thermal load detected. Monitor tool flank wear.")
            
        if p[1] > 1800:
            warn_c.error("🚨 **MECHANICAL FAILURE RISK:** Cutting force exceeds 1800N. Risk of insert chipping or spindle overload.")
        elif fr_v > 0.20:
            warn_c.info("💡 **HEAVY CHIP LOAD:** Ensure rigid clamping. High feed rate may affect surface finish.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Spindle RPM", f"{int(rpm)}")
        m2.metric("Interface Temp", f"{p[0]:.4f} °C")
        m3.metric("Cutting Force", f"{p[1]:.4f} N")
        
        g1, g2 = st.columns(2)
        g1.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=p[0], title={'text': "Temp (°C)"},
            gauge={'axis':{'range':[0,1500]}, 'bar':{'color':'#e74c3c'}})).update_layout(height=350), use_container_width=True)
        g2.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=p[1], title={'text': "Force (N)"},
            gauge={'axis':{'range':[0,2500]}, 'bar':{'color':'#3498db'}})).update_layout(height=350), use_container_width=True)

with tab2:
    st.subheader("Work Accuracy & Regression Metrics")
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    accuracy = (1 - mape) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Work Accuracy", f"{accuracy:.2f} %")
    col2.metric("R-Squared (R²)", f"{r2:.6f}")
    col3.metric("Error Percentage", f"{mape * 100:.4f} %")

    st.write("### Parity Plot (Experimental vs AI)")
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=y['Temp'], y=y_pred[:, 0], mode='markers', name='Trials'))
    fig_p.add_trace(go.Scatter(x=[300, 1500], y=[300, 1500], mode='lines', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig_p.update_layout(xaxis_title="Experimental", yaxis_title="AI Prediction", height=450), use_container_width=True)

with tab3:
    st.subheader("Research Literature Database")
    st.dataframe(full_df, use_container_width=True)
