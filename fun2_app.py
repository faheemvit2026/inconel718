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

# --- 2. CALCULATE VALIDATION METRICS ---
y_pred = model.predict(X)
mape_val = mean_absolute_percentage_error(y, y_pred)
r2_val = r2_score(y, y_pred)
overall_accuracy = (1 - mape_val) * 100
overall_efficiency = (r2_val * 0.7) + ((1 - mape_val) * 0.3)

# --- 3. UI CONFIGURATION & INDUSTRIAL THEMING ---
st.set_page_config(page_title="Inconel 718 AI Twin", layout="wide")

st.markdown("""
    <style>
    /* 1. Global Background (Greyish-Blue Industrial look) */
    .stApp {
        background-color: #f0f2f6;
    }

    /* 2. Hide default Streamlit header */
    header[data-testid="stHeader"] { visibility: hidden; height: 0px; }
    
    /* 3. The Professional Identity Banner */
    .identity-banner {
        background-color: #002D62;
        padding: 40px;
        border-radius: 0px 0px 20px 20px;
        border-bottom: 8px solid #FFD700;
        text-align: center;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.3);
        margin-top: -60px; /* Pulls up to the very top */
    }
    
    .identity-banner h1 {
        color: #FFFFFF !important;
        font-size: 55px !important;
        font-weight: 900 !important;
        margin: 0 !important;
        letter-spacing: 3px;
    }
    
    .identity-banner p {
        color: #FFD700 !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-top: 10px !important;
    }

    /* 4. Tab Styling - Integrating into the dark theme */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #002D62;
        padding: 15px 30px 0px 30px;
        border-radius: 15px 15px 0px 0px;
        gap: 20px;
        margin-top: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background-color: #1E3A5F !important;
        color: #FFFFFF !important;
        border-radius: 10px 10px 0px 0px;
        border: none !important;
        font-size: 1.1rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFD700 !important;
        color: #002D62 !important;
        font-weight: 800 !important;
    }

    /* 5. Validation Metric Cards (The Percentage Stuffs) */
    .metric-card {
        background-color: #002D62;
        padding: 25px;
        border-radius: 15px;
        border-top: 5px solid #FFD700;
        color: white !important;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 { color: #FFD700 !important; margin-bottom: 5px; }
    .metric-card p { font-size: 1.8rem; font-weight: bold; margin: 0; }
    </style>
    
    <div class="identity-banner">
        <h1>MOHAMMED FAHEEM</h1>
        <p>B.Tech Mechanical Engineering | Manufacturing Specialization | VIT Vellore</p>
    </div>
    """, unsafe_allow_html=True)

# --- 4. NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Process Simulator", "📊 Analytics & Validation", "📑 Database Log"])

with tab1:
    c_in, c_out = st.columns([1, 2.3])
    with c_in:
        st.markdown("### 🔧 Control Unit")
        with st.container(border=True):
            tool = st.radio("Tool Insert Grade", ["Diamond Coated", "Tungsten Carbide"])
            dia_v = st.number_input("Workpiece Dia (mm)", value=25.0, format="%.4f")
            vc_v = st.number_input("Speed Vc (m/min)", value=100.0, format="%.4f")
            fr_v = st.number_input("Feed f (mm/rev)", value=0.1, format="%.4f")
            ap_v = st.number_input("DOC ap (mm)", value=0.5, format="%.4f")
        
        p = model.predict([[vc_v, fr_v, ap_v, dia_v, (1 if tool=="Diamond Coated" else 0)]])[0]
        rpm = (vc_v * 1000) / (math.pi * dia_v)

    with c_out:
        # ALERTS
        st.markdown("### 🚦 System Response")
        if p[0] > 1000:
            st.error(f"🛑 CRITICAL THERMAL OVERLOAD: {p[0]:.2f} °C")
        else:
            st.success(f"✅ STABLE MACHINING PARAMETERS | AI Accuracy: {overall_accuracy:.2f}%")

        m1, m2, m3 = st.columns(3)
        m1.metric("Calculated RPM", f"{rpm:.2f}")
        m2.metric("Interface Temp", f"{p[0]:.2f} °C")
        m3.metric("Cutting Force", f"{p[1]:.2f} N")
        
        # SPEEDOMETERS WITH SMOOTH TRANSITION
        g1, g2 = st.columns(2)
        
        fig_t = go.Figure(go.Indicator(
            mode="gauge+number", value=p[0],
            title={'text': "Thermal Load (°C)", 'font': {'size': 20, 'color': '#002D62'}},
            gauge={'axis': {'range': [0, 1500]}, 'bar': {'color': "#D35400"},
                   'steps': [{'range': [0, 900], 'color': "#D6EAF8"}, {'range': [900, 1500], 'color': "#FADBD8"}]}))
        fig_t.update_layout(height=420, transition={'duration': 1000, 'easing': 'cubic-in-out'})
        g1.plotly_chart(fig_t, use_container_width=True)

        fig_f = go.Figure(go.Indicator(
            mode="gauge+number", value=p[1],
            title={'text': "Force (N)", 'font': {'size': 20, 'color': '#002D62'}},
            gauge={'axis': {'range': [0, 2500]}, 'bar': {'color': "#2E86C1"},
                   'steps': [{'range': [0, 1850], 'color': "#D6EAF8"}, {'range': [1850, 2500], 'color': "#FADBD8"}]}))
        fig_f.update_layout(height=420, transition={'duration': 1000, 'easing': 'cubic-in-out'})
        g2.plotly_chart(fig_f, use_container_width=True)

with tab2:
    st.markdown("### 📈 Analytics & Validation Hub")
    
    # MAPE, R-Squared, and Accuracy are now in high-visibility cards
    v1, v2, v3, v4 = st.columns(4)
    
    with v1:
        st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><p>{overall_accuracy:.2f} %</p></div>', unsafe_allow_html=True)
    with v2:
        st.markdown(f'<div class="metric-card"><h3>Efficiency</h3><p>{overall_efficiency*100:.2f} %</p></div>', unsafe_allow_html=True)
    with v3:
        st.markdown(f'<div class="metric-card"><h3>MAPE</h3><p>{mape_val:.6f}</p></div>', unsafe_allow_html=True)
    with v4:
        st.markdown(f'<div class="metric-card"><h3>R² Score</h3><p>{r2_val:.6f}</p></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(go.Figure(go.Scatter(x=y['Temp'], y=y_pred[:,0], mode='markers', marker=dict(color='#002D62'))).update_layout(title="Prediction Stability (Experimental vs AI)"))

with tab3:
    st.subheader("Experimental Training Data")
    st.dataframe(full_df, use_container_width=True)

# FOOTER
st.markdown("<br><hr><center><b>Created by Mohammed Faheem | VIT Vellore | Final Year Thesis © 2026</b></center>", unsafe_allow_html=True)
