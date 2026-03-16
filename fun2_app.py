# Add these lines into your existing metric section
from sklearn.metrics import mean_absolute_error

y_pred_all = rf.predict(X)
mae = mean_absolute_error(y, y_pred_all)
accuracy = 100 - (mae / y.mean() * 100)

# Display in the dashboard
st.metric("Model Accuracy", f"{accuracy:.2f}%")
st.metric("Avg Prediction Error", f"{mae:.2f} °C")
