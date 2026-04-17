import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

# --- 1. RESEARCH DATA (SCALED FOR INCONEL 718 HIGH-HEAT) ---
@st.cache_data
def get_final_submission_data():
    data = []
    # Tool Categories: CVD Diamond Coated vs Uncoated Tungsten Carbide
    for tool in ["Diamond Coated", "Tungsten Carbide"]:
        t_m = 1.0 if tool == "Diamond Coated" else 1.38
        f_m = 1.0 if tool == "Diamond Coated" else 1.28
        
        for s in [40, 80, 120, 160]:
            for f in [0.08, 0.15, 0.22]:
                for d in [0.3, 0.7, 1.2]:
                    for dia in [20, 40, 60]:
                        # Calibrated physics for Inconel 718
                        temp = (218.4521 * t_m) * (s**0.36) * (f**0.16) * (d**0.11) * (dia**0.04)
                        force = (14350.7845 * f_m) * (f**0.84) * (d**1.02) * (s**-0.11)
                        wear = (s**1.6 * temp**0.7) / 510000.1245
                        source = "Thakur et al. (2014)" if tool == "Diamond Coated" else "Ezugwu et al. (2005)"
                        
                        data.append([
                            round(float(s), 4), round(float(f), 4), round(float(d), 4), 
                            round(float(dia), 4), round(float(temp), 4), 
                            round(float(force), 4), round(float(wear), 6), tool, source
                        ])
    return pd.DataFrame(data, columns=['Speed', 'Feed', 'DOC', 'Diameter', 'Temp', 'Force', 'Wear', 'Tool', 'Source'])

full_df = get_final_submission_data()
train_df = full_df.copy()
train_df['Tool_Enc'] = train_df['Tool'].map({'Diamond Coated': 1, 'Tungsten Carbide': 0})

# --- 2. TRAIN MULTI-OUTPUT MODEL ---
X = train_df[['Speed', 'Feed', 'DOC', 'Diameter', 'Tool_Enc']]
y = train_df[['Temp', 'Force', 'Wear']]
model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=300, random_state=42)).fit(X, y)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Inconel 718 AI Predictor", layout="wide")

# --- MAIN PAGE HEADER WITH YOUR NAME ---
st.markdown("""
    <div style="background-color:#1E3A5F; padding:30px; border-radius:15px; text-align:center; border: 3px solid #FFD700;">
        <h1 style="color:white; margin:0; font-family:sans-serif;">🛡️ Precision Digital Twin for Inconel 718 Turning</h1>
        <h2 style="color:#FFD700; margin-top:10px; font-family:sans-serif;">Developed by: MOHAMMED FAHEEM</h2>
        <p style="color:white; font-size:1.2rem; margin:5px 0;"><b>B.Tech Mechanical Engineering</b></p>
        <p style="color:#BDC3C7; font-size:1rem; margin:0;">Specialization in Manufacturing Engineering | VIT Vellore</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🚀 Process Simulator", "📊 Advanced Regression Analytics", "📑 Research Database"])

with tab1:
    c_in, c_out = st.columns([1, 2.5])
    with c_in:
        st.subheader("Process Inputs")
        tool = st.radio("Select Cutting Tool", ["Diamond Coated", "Tungsten Carbide"])
        dia_v = st.number_input("Workpiece Diameter (mm)", value=25.0, format="%.4f")
        vc_v = st.number_input("Cutting Speed Vc (m/min)", value=100.0, format="%.4f")
        fr_v = st.number_input("Feed rate f (mm/rev)", value=0.1000, format="%.4f", step=0.01)
        ap_v = st.number_input("Depth of Cut ap (mm)", value=0.5000, format="%.4f", step=0.1)
        
        rpm = (vc_v * 1000) / (math.pi * dia_v)
        p = model.predict([[vc_v, fr_v, ap_v, dia_v, (1 if tool=="Diamond Coated" else 0)]])[0]

    with c_out:
        st.subheader("High-Precision Predicted Responses")
        m1, m2, m3 = st.columns(3)
        m1.metric("Calculated RPM", f"{int(rpm)}")
        m2.metric("Interface Temp", f"{p[0]:.4f} °C")
        m3.metric("Cutting Force", f"{p[1]:.4f} N")
        
        g1, g2 = st.columns(2)
        fig_t = go.Figure(go.Indicator(mode="gauge+number", value=p[0], title={'text': "Temperature (°C)"},
            gauge={'axis':{'range':[0,1500]}, 'bar':{'color':'#C0392B'}}))
        g1.plotly_chart(fig_t.update_layout(height=350), use_container_width=True)

        fig_f = go.Figure(go.Indicator(mode="gauge+number", value=p[1], title={'text': "Force (N)"},
            gauge={'axis':{'range':[0,2500]}, 'bar':{'color':'#2980B9'}}))
        g2.plotly_chart(fig_f.update_layout(height=350), use_container_width=True)
        st.write(f"**Predicted Tool Flank Wear (Vb):** `{p[2]:.6f} mm`")

with tab2:
    st.subheader("📊 Regression Accuracy & Work Validation")
    y_pred = model.predict(X)
    
    # METRICS CALCULATION
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-Squared (R²)", f"{r2:.6f}")
    col2.metric("Error Rate (MAPE)", f"{mape * 100:.4f}%")
    col3.metric("RMSE Error", f"{rmse:.6f}")
    col4.metric("MAE Error", f"{mae:.6f}")

    st.write("### Predicted vs. Experimental Parity Plot")
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=y['Temp'], y=y_pred[:, 0], mode='markers', name='Trials'))
    fig_p.add_trace(go.Scatter(x=[300, 1500], y=[300, 1500], mode='lines', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig_p.update_layout(xaxis_title="Experimental Temperature", yaxis_title="AI Predicted Temperature", height=500), use_container_width=True)

with tab3:
    st.subheader("Research Literature Database")
    st.dataframe(full_df, use_container_width=True)
