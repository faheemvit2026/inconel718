import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# --- 1. RESEARCH DATA ENGINE ---
@st.cache_data
def get_hybrid_dual_data():
    np.random.seed(42)
    
    # YOUR EXPERIMENTAL DATA - LOCKED TO DIAMOND COATED (Material 1)
    # Format: [Vc, Feed, DOC, Dia, Material_Code (1=DCC), Temp, Force]
    dcc_experimental = [
        [40, 0.08, 0.25, 32, 1, 350.0, 510.5], 
        [40, 0.10, 0.25, 32, 1, 411.2, 496.5], 
        [55, 0.08, 0.25, 32, 1, 544.2, 345.4], 
        [55, 0.10, 0.25, 32, 1, 644.5, 400.9], 
        [60, 0.08, 0.25, 32, 1, 670.4, 290.4], 
        [60, 0.10, 0.25, 32, 1, 695.0, 288.0]  
    ]
    
    # 215 RESEARCH POINTS - For general WC and DCC behavior
    research_pts = []
    for _ in range(215):
        vc = np.random.uniform(30, 150)
        f = np.random.uniform(0.05, 0.20)
        ap = np.random.uniform(0.1, 0.8)
        dia = np.random.uniform(10, 50)
        mat = np.random.choice([0, 1]) # 0=WC, 1=DCC
        
        # Physics Logic: WC (0) runs hotter than DCC (1)
        # Force is usually higher for WC due to higher friction coefficient
        temp = (vc * 7.5) + (f * 1800) + (ap * 120) - (mat * 60) + np.random.normal(0, 8)
        force = (ap * 1100) + (f * 1400) - (vc * 0.4) + (dia * 1.5) + (mat * -30)
        research_pts.append([vc, f, ap, dia, mat, temp, force])
        
    combined = dcc_experimental + research_pts
    return pd.DataFrame(combined, columns=['Speed', 'Feed', 'DOC', 'Diameter', 'Material', 'Temp', 'Force'])

df = get_hybrid_dual_data()
X = df[['Speed', 'Feed', 'DOC', 'Diameter', 'Material']]
y = df[['Temp', 'Force']]

# Higher estimators for better anchoring to the DCC experimental points
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=400, random_state=42)).fit(X, y)

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="Inconel 718 AI Twin", layout="wide")

st.markdown(f"""
    <style>
    .stApp {{ background-color: #0E1117; color: #E0E0E0; }}
    header[data-testid="stHeader"] {{ visibility: hidden; height: 0px; }}
    .identity-banner {{
        background-color: #1A1C24; padding: 25px; border-bottom: 5px solid #FFD700;
        text-align: center; margin-top: -60px;
    }}
    .identity-banner h1 {{ color: #FFFFFF !important; font-size: 40px !important; margin: 0 !important; }}
    .metric-card {{ background-color: #1A1C24; padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700; text-align: center; }}
    </style>
    <div class="identity-banner">
        <h1>MOHAMMED FAHEEM</h1>
        <p>Mechanical Engineering | Manufacturing AI | DCC Experimental Calibration</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🚀 AI Predictor", "📊 Validation Metrics"])

with tab1:
    c_in, c_out = st.columns([1, 2.3])
    with c_in:
        st.subheader("Process Parameters")
        tool_choice = st.radio("Tool Insert Material:", ["Tungsten Carbide (WC)", "Diamond Coated (DCC)"])
        mat_idx = 1 if tool_choice == "Diamond Coated (DCC)" else 0
        
        dia_v = st.number_input("Workpiece Dia (mm)", value=32.0, step=1.0)
        vc_v = st.number_input("Cutting Speed (Vc)", value=40.0, step=5.0)
        fr_v = st.number_input("Feed Rate (f)", value=0.080, format="%.3f", step=0.01)
        ap_v = st.number_input("Depth of Cut (ap)", value=0.25, step=0.05)
        
        rpm = (vc_v * 1000) / (math.pi * dia_v)
        p = model.predict([[vc_v, fr_v, ap_v, dia_v, mat_idx]])[0]

    with c_out:
        st.subheader(f"AI Prediction: {tool_choice}")
        
        # Display different status based on tool limits
        if mat_idx
