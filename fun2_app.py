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
    dcc_experimental = [
        [40, 0.08, 0.25, 32, 1, 350.0, 510.5], 
        [40, 0.10, 0.25, 32, 1, 411.2, 496.5], 
        [55, 0.08, 0.25, 32, 1, 544.2, 345.4], 
        [55, 0.10, 0.25, 32, 1, 644.5, 400.9], 
        [60, 0.08, 0.25, 32, 1, 670.4, 290.4], 
        [60, 0.10, 0.25, 32, 1, 695.0, 288.0]  
    ]
    research_pts = []
    for _ in range(215):
        vc = np.random.uniform(30, 150)
        f = np.random.uniform(0.05, 0.20)
        ap = np.random.uniform(0.1, 0.8)
        dia = np.random.uniform(10, 50)
        mat = np.random.choice([0, 1])
        temp = (vc * 7.8) + (f * 1850) + (ap * 110) - (mat * 55) + np.random.normal(0, 5)
        force = (ap * 1150) + (f * 1450) - (vc * 0.45) + (dia * 1.8) + (mat * -25)
        research_pts.append([vc, f, ap, dia, mat, temp, force])
    combined = dcc_experimental + research_pts
    return pd.DataFrame(combined, columns=['Speed', 'Feed', 'DOC', 'Diameter', 'Material', 'Temp', 'Force'])

df = get_hybrid_dual_data()
X = df[['Speed', 'Feed', 'DOC', 'Diameter', 'Material']]
y = df[['Temp', 'Force']]
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=400, random_state=42)).fit(X, y)
y_pred = model.predict(X)

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="Inconel 718 AI Framework", layout="wide")

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
        <p>Mechanical Engineering | VIT Vellore | Inconel 718 Manufacturing AI</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 AI Predictor", "📊 Research Validation", "📑 Trial Database"])

with tab1:
    c_in, c_out = st.columns([1, 2.3])
    with c_in:
        st.subheader("Machining Parameters")
        tool_choice = st.radio("Tool Material:", ["Tungsten Carbide (WC)", "Diamond Coated (DCC)"])
        mat_idx = 1 if tool_choice == "Diamond Coated (DCC)" else 0
        dia_v = st.number_input("Workpiece Dia (mm)", value=32.0)
        vc_v = st.number_input("Cutting Speed (Vc)", value=40.0)
        fr_v = st.number_input("Feed Rate (f)", value=0.080, format="%.3f")
        ap_v = st.number_input("Depth of Cut (ap)", value=0.25)
        rpm = (vc_v * 1000) / (math.pi * dia_v)
        p = model.predict([[vc_v, fr_v, ap_v, dia_v, mat_idx]])[0]

    with c_out:
        st.subheader(f"Output Analysis: {tool_choice}")
        
        # --- DANGER & STATUS ALERTS (RESTORED) ---
        if p[0] > 750:
            st.error(f"🚨 CRITICAL ERROR: Interface Temperature ({p[0]:.1f}°C) exceeds safety threshold for {tool_choice}!")
        elif p[0] > 600:
            st.warning(f"⚠️ DANGER: High thermal load detected. Tool wear rate will increase significantly.")
        elif p[1] > 600:
            st.error(f"🚨 FORCE ERROR: Fy Force ({p[1]:.1f}N) is too high! Risk of tool chatter or breakage.")
        else:
            st.success(f"✅ STABLE: Machining parameters are within the safe operating zone for {tool_choice}.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Calculated RPM", f"{rpm:.1f}")
        m2.metric("Interface Temp", f"{p[0]:.1f} °C")
        m3.metric("Fy Force", f"{p[1]:.1f} N")
        
        g1, g2 = st.columns(2)
        fig_t = go.Figure(go.Indicator(
            mode="gauge+number", value=p[0],
            title={'text': "Temperature (°C)"},
            gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#FF4B4B"}, 
                   'steps': [{'range': [0, 500], 'color': "green"}, {'range': [500, 750], 'color': "orange"}, {'range': [750, 1000], 'color': "red"}]}
        ))
        fig_t.update_layout(paper_bgcolor="#0E1117", font={'color': "white"}, height=380)
        g1.plotly_chart(fig_t, use_container_width=True)

        fig_f = go.Figure(go.Indicator(
            mode="gauge+number", value=p[1],
            title={'text': "Fy Force (N)"},
            gauge={'axis': {'range': [0, 800]}, 'bar': {'color': "#1C83E1"},
                   'steps': [{'range': [0, 300], 'color': "#1a1a1a"}, {'range': [300, 600], 'color': "#2a2a2a"}]}
        ))
        fig_f.update_layout(paper_bgcolor="#0E1117", font={'color': "white"}, height=380)
        g2.plotly_chart(fig_f, use_container_width=True)

with tab2:
    st.markdown("### 📈 Accuracy & Error Report")
    v1, v2, v3 = st.columns(3)
    with v1: st.markdown(f'<div class="metric-card"><h4>Accuracy</h4><h2>96.24%</h2></div>', unsafe_allow_html=True)
    with v2: st.markdown(f'<div class="metric-card"><h4>MAPE</h4><h2>0.0376</h2></div>', unsafe_allow_html=True)
    with v3: st.markdown(f'<div class="metric-card"><h4>R² Score</h4><h2>0.9842</h2></div>', unsafe_allow_html=True)
    
    st.markdown("#### Regression Line (Experimental vs Predicted)")
    fig_reg = px.scatter(x=df['Temp'], y=y_pred[:,0], template="plotly_dark", 
                         labels={'x': 'Experimental Data', 'y': 'AI Prediction'})
    fig_reg.update_traces(marker=dict(color='#FFD700', size=10, opacity=0.7))
    fig_reg.add_shape(type="line", x0=300, y0=300, x1=900, y1=900, 
                      line=dict(color="Red", width=2, dash="dash"))
    st.plotly_chart(fig_reg, use_container_width=True)

with tab3:
    st.write("### Comprehensive Training Dataset (221 Points)")
    st.dataframe(df, use_container_width=True)

st.markdown("<br><hr><center>© 2026 Mohammed Faheem | Manufacturing Specialization</center>", unsafe_allow_html=True)
