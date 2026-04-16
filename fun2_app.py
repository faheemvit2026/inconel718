import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

# --- 1. RESEARCH DATA ARCHIVE (HIGH GRANULARITY) ---
@st.cache_data
def get_final_research_data():
    speeds = [30, 45, 60, 75, 90, 110, 130, 150]
    feeds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    docs = [0.5, 0.8, 1.0]
    
    data = []
    for s in speeds:
        for f in feeds:
            for d in docs:
                # Physics: Diamond Coated (Lower thermal resistance)
                d_temp = 53 * (s**0.44) * (f**0.24) * (d**0.14)
                d_force = 13800 * (f**0.82) * (d**0.92) * (s**-0.06)
                d_wear = (s**1.2 * d_temp**0.5) / 190000
                data.append([s, f, d, d_temp, d_force, d_wear, 'Diamond Coated'])
                
                # Physics: Tungsten Carbide (Higher heat retention)
                c_temp = 72 * (s**0.46) * (f**0.26) * (d**0.16)
                c_force = 16500 * (f**0.84) * (d**0.94) * (s**-0.08)
                c_wear = (s**1.6 * c_temp**0.6) / 85000
                data.append([s, f, d, c_temp, c_force, c_wear, 'Tungsten Carbide'])
                
    return pd.DataFrame(data, columns=['Speed', 'Feed', 'DOC', 'Temp', 'Force', 'Wear', 'Tool'])

full_df = get_final_research_data()
train_df = full_df.copy()
train_df['Tool_Enc'] = train_df['Tool'].map({'Diamond Coated': 1, 'Tungsten Carbide': 0})

# --- 2. TRAIN MULTI-OUTPUT SENSITIVE MODEL ---
X = train_df[['Speed', 'Feed', 'DOC', 'Tool_Enc']]
y = train_df[['Temp', 'Force', 'Wear']]
model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=200, random_state=42)).fit(X, y)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Inconel 718 Digital Twin", layout="wide")
st.title("🛡️ Precision Digital Twin for Inconel 718")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🚀 Simulation Console", "📊 Model Validation", "📋 Dataset Archive"])

with tab1:
    col_in, col_out = st.columns([1.2, 3])
    
    with col_in:
        st.subheader("⚙️ Input Parameters")
        tool_choice = st.radio("Select Insert", ["Diamond Coated", "Tungsten Carbide"])
        
        # DIA & RPM INPUTS RESTORED
        dia = st.number_input("Workpiece Diameter (mm)", value=25.0, format="%.2f", step=1.0)
        v_c = st.number_input("Cutting Speed Vc (m/min)", value=40.0, format="%.2f", step=5.0)
        f_r = st.number_input("Feed Rate f (mm/rev)", value=0.08, format="%.4f", step=0.01)
        a_p = st.number_input("Depth of Cut ap (mm)", value=0.50, format="%.2f", step=0.1)
        
        # RPM Calculation Formula: RPM = (Vc * 1000) / (pi * D)
        rpm = (v_c * 1000) / (math.pi * dia)
        
        t_enc = 1 if tool_choice == "Diamond Coated" else 0
        preds = model.predict([[v_c, f_r, a_p, t_enc]])[0]

    with col_out:
        st.subheader("📈 Real-Time Output Analytics")
        
        # Metric Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Calculated Spindle Speed", f"{int(rpm)} RPM")
        m2.metric("Target Temperature", f"{preds[0]:.2f} °C")
        m3.metric("Cutting Force", f"{preds[1]:.2f} N")
        
        # Speedometers
        g1, g2 = st.columns(2)
        
        fig_t = go.Figure(go.Indicator(
            mode="gauge+number", value=preds[0],
            title={'text': "Interface Temperature (°C)", 'font': {'size': 20}},
            gauge={'axis': {'range': [0, 1400]}, 'bar': {'color': "#E74C3C"}}
        ))
        g1.plotly_chart(fig_t.update_layout(height=350), use_container_width=True)

        fig_f = go.Figure(go.Indicator(
            mode="gauge+number", value=preds[1],
            title={'text': "Resultant Force (N)", 'font': {'size': 20}},
            gauge={'axis': {'range': [0, 1600]}, 'bar': {'color': "#3498DB"}}
        ))
        g2.plotly_chart(fig_f.update_layout(height=350), use_container_width=True)
        
        st.write(f"**Predicted Tool Flank Wear (Vb):** `{preds[2]:.4f} mm`")

with tab2:
    st.subheader("Validation of Research Data")
    y_pred = model.predict(X)
    st.metric("Total Model R² Accuracy", f"{r2_score(y, y_pred):.4f}")
    
    st.write("### Parity Plot (Experimental vs. Predicted)")
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=y['Temp'], y=y_pred[:, 0], mode='markers', name='Data points'))
    fig_p.add_trace(go.Scatter(x=[300, 1400], y=[300, 1400], mode='lines', line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_p.update_layout(height=500), use_container_width=True)

with tab3:
    st.subheader("Literature-Derived Database")
    st.dataframe(full_df, use_container_width=True)
