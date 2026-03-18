# --- UPDATE: DATA GENERATION (10 to 350 m/min) ---
@st.cache_data
def generate_research_data():
    num_samples = 200 # Increased samples for better low-end resolution
    # Speed: 10 m/min (Ultra-low) to 350 m/min (Extreme High-speed)
    speeds = np.geomspace(10, 350, num_samples)
    feeds = np.tile(np.linspace(0.04, 0.3, 20), 10)
    docs = np.repeat(np.linspace(0.15, 2.5, 10), 20)

    def get_targets(s, f, d):
        # Physics adjusted for ultra-low speed rubbing effects
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

# --- UPDATE: SIDEBAR SLIDERS ---
with col_in:
    st.subheader("🕹️ Parameters")
    dia = st.number_input("Workpiece Dia (mm)", 25.0, format="%.4f")
    
    # Range now starts at 10.0 instead of 40 or 75
    v_c = st.slider("Cutting Speed (m/min)", min_value=10.0, max_value=350.0, value=10.0, step=1.0)
    f_r = st.slider("Feed Rate (mm/rev)", min_value=0.04, max_value=0.3, value=0.04, step=0.01)
    a_p = st.slider("Depth of Cut (mm)", min_value=0.15, max_value=2.5, value=0.15, step=0.05)
