import streamlit as st  # 1. IMPORT FIRST
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# 2. CONFIG SECOND
st.set_page_config(page_title="Inconel 718 Precision Hub", layout="wide")

# 3. CACHED FUNCTIONS THIRD
@st.cache_data
def generate_research_data():
    num_samples = 200 
    # Speed: 10 m/min (Ultra-low) to 350 m/min (High-speed)
    speeds = np.geomspace(10, 350, num_samples)
    feeds = np.tile(np.linspace(0.04, 0.3, 20), 10)
    docs = np.repeat(np.linspace(0.15, 2.5, 10), 20)

    def get_targets(s, f, d):
        t = 160 + (18 * s**0.72) + (230 * f**0.45) + (90 * d**0.3)
        fy = (2050 * f**0.78 * d**0.95) + (s * 0.15)
        fx = fy * (0.44 + (0.04 * (s/350)))
        fz = fy * (0.64 + (0.08 * (s/350)))
        vb = (0.00008 * s**1.92) + (0.065 * f) + (0.014 * d)
        return [t, fx, fy, fz, vb]

    results = [get_targets(s, f, d) for s, f, d in zip(speeds, feeds, docs)]
    df_res = pd.DataFrame(results, columns=['Temp', 'Fx', 'Fy', 'Fz', 'Vb'])
    df_res['Speed'], df_res['Feed'], df_res['DOC'] = speeds, feeds, docs
    return df_res

# 4. DATA AND MODEL INITIALIZATION
df = generate_research_data()
X = df[['Speed', 'Feed', 'DOC']]
y = df[['Temp', 'Fx', 'Fy', 'Fz', 'Vb']]

@st.cache_resource
def train_model(X_train, y_train):
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=250, random_state=42))
    return model.fit(X_train, y_train)

model = train_model(X, y)
y_pred = model.predict(X)

# 5. UI ELEMENTS (TABS AND SLIDERS)
tab1, tab2, tab3 = st.tabs(["🎮 Dashboard", "📈 Validation", "📋 Raw Data"])

with tab1:
    st.title("🛡️ Inconel 718: Multi-Variable Machining Hub")
    st.markdown("Developed by **Mohammed Faheem M S** | Range: **10 - 350 m/min**")
    
    col_in, col_out = st.columns([1, 3])
    
    with col_in:
        st.subheader("🕹️ Parameters")
        dia = st.number_input("Workpiece Dia (mm)", 25.0, format="%.4f")
        
        # Ranges starting from 10.0 for ultra-low end
        v_c = st.slider("Cutting Speed (m/min)", 10.0, 350.0, 10.0, 1.0)
        f_r = st.slider("Feed Rate (mm/rev)", 0.04, 0.3, 0.04, 0.01)
        a_p = st.slider("Depth of Cut (mm)", 0.15, 2.5, 0.15, 0.05)
        
        rpm = (1000 * v_c) / (math.pi * dia)
        p = model.predict([[v_c, f_r, a_p]])[0]

    with col_out:
        # Display Gauges and Metrics here (same as previous working code)
        m1, m2, m3 = st.columns(3)
        m1.metric("Temperature", f"{p[0]:.2f} °C")
        m2.metric("Main Force (Fy)", f"{p[2]:.2f} N")
        m3.metric("Flank Wear", f"{p[4]:.4f} mm")
        
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=p[0], 
                title={'text': "Thermal Load (°C)"}, gauge={'axis': {'range': [0, 1300]}, 'bar': {'color': "#ff4b4b"}})), use_container_width=True)
        with g2:
            st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=p[2], 
                title={'text': "Cutting Force (N)"}, gauge={'axis': {'range': [0, 2500]}, 'bar': {'color': "#0068c9"}})), use_container_width=True)
