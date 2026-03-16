from sklearn.metrics import mean_squared_error, r2_score

# --- CALCULATIONS ---
mse = mean_squared_error(y, y_pred_all)
r2 = r2_score(y, y_pred_all)

# --- DASHBOARD LAYOUT (Updated Section) ---
st.subheader("📊 Statistical Reliability & Model Fit")
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("R² Score", f"{r2:.4f}")
    st.caption("Proportion of variance explained (1.0 is perfect).")

with k2:
    st.metric("Mean Squared Error", f"{mse:.2f}")
    st.caption("Average squared difference (Lower is better).")

with k3:
    st.metric("Error Percentage (MAPE)", f"{mape:.2f}%")
    st.caption("Relative average deviation.")

with k4:
    st.metric("Model Accuracy", f"{accuracy_pct:.2f}%")
    st.caption("Overall prediction reliability.")
