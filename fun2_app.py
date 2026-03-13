import streamlit as st
import plotly.graph_objects as go

# --- CORE PREDICTION LOGIC ---
def predict_inconel_temp(vc, feed, mode):
    mql_factor = 1 if mode == "MQL Supply" else 0
    # Engineering formula for Inconel 718
    temp = 165 + (7.5 * vc) + (950 * feed) - (75 * mql_factor)
    return temp

# --- PAGE CONFIG ---
st.set_page_config(page_title="Inconel AI Dashboard", layout="wide")

# --- WATERMARK & THEME CSS ---
st.markdown(f"""
    <style>
    .watermark {{
        position: fixed; bottom: 15px; right: 25px; opacity: 0.25;
        font-size: 22px; color: #555; z-index: 1000; pointer-events: none;
        font-family: sans-serif; font-weight: bold;
    }}
    .stMetric {{ background-color: #1e2130; padding: 20px; border-radius: 12px; }}
    </style>
    <div class="watermark">mdfaheem</div>
    """, unsafe_allow_html=True)

st.title("🛡️ Inconel 718 Thermal Command Center")
st.caption("Developed by mdfaheem | Industrial AI Predictor")
st.divider()

# --- INPUTS (Sidebar) ---
st.sidebar.header("🕹️ CONTROL PANEL")
vc = st.sidebar.slider("Cutting Speed (Vc)", 40.0, 70.0, 55.0)
feed = st.sidebar.slider("Feed Rate (f)", 0.05, 0.20, 0.10, step=0.01)
mode = st.sidebar.radio("Cooling State", ["Dry Turning", "MQL Supply"])
safety_limit = 650.0

# Calculation
temp_result = predict_inconel_temp(vc, feed, mode)

# --- ANALOG & DIGITAL LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    # ANALOG GAUGE (Plotly)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = temp_result,
        title = {'text': "Analog Temperature (°C)"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [300, 850], 'tickwidth': 1},
            'bar': {'color': "#00ffcc"},
            'steps': [
                {'range': [300, 550], 'color': "#1a472a"},
                {'range': [550, 700], 'color': "#47411a"},
                {'range': [700, 850], 'color': "#471a1a"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': safety_limit}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔢 Digital System Readout")
    
    # DIGITAL METRIC
    st.metric(label="Calculated Heat Output", value=f"{temp_result:.1f} °C", 
              delta=f"{safety_limit - temp_result:.1f} to Limit", delta_color="inverse")
    
    # STATUS CARDS
    if temp_result > safety_limit:
        st.error(f"ALERT: CRITICAL HEAT detected for Inconel 718.")
    else:
        st.success("SYSTEM STATUS: NOMINAL")

    st.info(f"**Parameters:** {vc} m/min @ {feed} mm/rev ({mode})")
    
    # EXTRA ANALYTICS
    with st.expander("View Engineering Metadata"):
        st.write(f"**Developer:** mdfaheem")
        st.write(f"**Material:** Inconel 718 (Nickel-based Superalloy)")
        st.write(f"**MQL Efficiency:** ~12.5% reduction