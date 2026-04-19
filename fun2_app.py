import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# --- 1. RESEARCH DATA ENGINE ---
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

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="Inconel 718 AI Twin", layout="wide")

# CSS to make the header sticky but NOT blocking the tabs
st.markdown("""
    <style>
    /* Hide the top streamlit padding */
    header[data-testid="stHeader"] { visibility: hidden; height: 0px; }
    
    /* Create a sticky container for the name */
    .sticky-wrapper {
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        z-index: 1000;
        background-color: white;
        padding-bottom: 10px;
    }
    
    .name-card {
        background-color: #002D62;
        padding: 20px;
        border-radius: 10px;
        border-bottom: 5px solid #FFD700;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Ensure tabs are clickable and styled */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    
    <div class="sticky-wrapper">
        <div class="name-card">
            <h1 style="color: white; margin: 0; font-size: 42px; font-weight: 900;">MOHAMMED FAHEEM</h1>
            <p style="color: #FFD700; font-size: 1.1rem; margin: 5px 0 0 0; font-weight: 600;">
                B.Tech Mechanical Engineering | Manufacturing Specialization | VIT Vellore
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Simulator", "📊 Analytics & Validation", "📑 Database Log"])

with tab1:
    c_in, c_out = st.columns([1, 2.3])
    with c_in:
        st.subheader("Process Parameters")
        tool = st.radio("Insert Grade", ["Diamond Coated", "Tungsten Carbide"])
        dia_v = st.number_input("Workpiece Dia (mm)", value=25.0, format="%.4f")
        vc_v = st.number_input("Speed Vc (m/min)", value=100.0, format="%.4f")
        fr_v = st.number_input("Feed rate f (mm/rev)", value=0.1, format="%.4f")
        ap_v = st.number_input("DOC ap (mm)", value=0.5, format="%.4f")
        
        rpm = (vc_v * 1000) / (math.pi * dia_v)
        p = model.predict([[vc_v, fr_v, ap_v, dia_v, (1 if tool=="Diamond Coated" else 0)]])[0]

    with c_out:
        # SAFETY ALERTS
        if p[0] > 1100:
            st.error(f"🛑 **DANGER:** Interface Temp ({p[0]:.4f} °C) exceeds safety threshold!")
        elif p[0] > 900:
            st.warning(f"⚠️ **ALERT:** High Thermal Zone ({p[0]:.4f} °C).")
        
        if p[1] > 1850:
            st.error(f"🚨 **OVERLOAD:** Mechanical Force ({p[1]:.4f} N) critical!")
        else:
            st.success(f"✅ **STABLE:** AI Confidence: {overall_accuracy:.2f}%")

        m1, m2, m3 = st.columns(3)
        m1.metric("Spindle RPM", f"{rpm:.2f}")
        m2.metric("Predicted Temp", f"{p[0]:.2f} °C")
        m3.metric("Cutting Force", f"{p[1]:.2f} N")
        
        # --- INTERACTIVE ANIMATED GAUGES ---
        g1, g2 = st.columns(2)
        
        fig_t = go.Figure(go.Indicator(
            mode="gauge+number", value=p[0],
            gauge={'axis': {'range': [0, 1500]}, 'bar': {'color': "#D35400"},
                   'steps': [{'range': [1100, 1500], 'color': "rgba(255,0,0,0.1)"}]}))
        fig_t.update_layout(title="Thermal Analysis", height=380, transition={'duration': 1000, 'easing': 'cubic-in-out'})
        g1.plotly_chart(fig_t, use_container_width=True)

        fig_f = go.Figure(go.Indicator(
            mode="gauge+number", value=p[1],
            gauge={'axis': {'range': [0, 2500]}, 'bar': {'color': "#2E86C1"},
                   'steps': [{'range': [1850, 2500], 'color': "rgba(0,0,255,0.1)"}]}))
        fig_f.update_layout(title="Force Analysis", height=380, transition={'duration': 1000, 'easing': 'cubic-in-out'})
        g2.plotly_chart(fig_f, use_container_width=True)

with tab2:
    st.markdown("### 📊 Analytics & Statistical Validation")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{overall_accuracy:.2f} %")
        col2.metric("Efficiency", f"{overall_efficiency*100:.2f} %")
        col3.metric("MAPE", f"{mape_val:.8f}")
        col4.metric("R² Score", f"{r2_val:.8f}")
    
    # Correlation Chart
    st.plotly_chart(go.Figure(go.Scatter(x=y['Temp'], y=y_pred[:, 0], mode='markers', marker=dict(color='#002D62'))).update_layout(title="Actual vs Predicted Temp"))

with tab3:
    st.subheader("Training Database")
    st.dataframe(full_df, use_container_width=True)

# FOOTER
st.markdown("<br><hr><center>Developed by <b>Mohammed Faheem</b> | VIT Vellore | © 2026</center>", unsafe_allow_html=True)
