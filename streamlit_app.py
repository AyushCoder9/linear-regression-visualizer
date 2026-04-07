import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

# Use the full width of the screen
st.set_page_config(page_title="Linear Regression Visualizer", page_icon="📈", layout="wide")

# Customizing the title and applying some custom HTML/CSS
st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h1 style='color: #4B4B4B;'>📈 Interactive Linear Regression Visualizer</h1>
        <p style='font-size: 18px; color: #666;'>Dynamically explore how linear models fit differently dispersed data.</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# Application state for animations
if 'run_animation' not in st.session_state:
    st.session_state.run_animation = False

# Sidebar for inputs
st.sidebar.title("App Settings")
st.sidebar.markdown("Configure the **Data Generation** parameters.")

n_samples = st.sidebar.slider("Number of samples (N)", min_value=10, max_value=500, value=150, step=10)
noise_level = st.sidebar.slider("Noise level (Variance)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)

st.sidebar.markdown("Configure the **True Linear Relation** parameters.")
true_slope = st.sidebar.slider("True Slope (m)", min_value=-10.0, max_value=10.0, value=3.0, step=0.5)
true_intercept = st.sidebar.slider("True Intercept (c)", min_value=-50.0, max_value=50.0, value=5.0, step=1.0)

# Generate synthetic dataset
@st.cache_data
def generate_data(n, slope, intercept, noise):
    np.random.seed(42) # Consistent random points distribution for simplicity
    X = np.linspace(-10, 10, n).reshape(-1, 1)
    
    # Adding noise to line y = mx + c
    y_true = slope * X.flatten() + intercept
    y_noisy = y_true + np.random.normal(0, noise, n)
    return X, y_noisy, y_true

X, y, y_true = generate_data(n_samples, true_slope, true_intercept, noise_level)

# Train tracking & displaying
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Layout: Split into top metrics and bottom graph
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(label="Calculated Slope", value=f"{model.coef_[0]:.2f}", delta=f"{model.coef_[0] - true_slope:.2f} (err)")
with c2:
    st.metric(label="Calculated Intercept", value=f"{model.intercept_:.2f}", delta=f"{model.intercept_ - true_intercept:.2f} (err)")
with c3:
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.1f}")
with c4:
    st.metric(label="R² Score (Accuracy)", value=f"{r2:.3f}")

st.divider()

# Chart rendering
st.subheader("Data Scatter vs Regression Line Forecast")

# Using Matplotlib to stylistically display the lines 
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the noisy data points
ax.scatter(X, y, color='#2c3e50', alpha=0.6, edgecolors='w', s=50, label='Generated Data')

# Plot the ideal line without noise
ax.plot(X, y_true, color='#27ae60', linestyle='--', linewidth=2, label='True Relation Line')

# Plot the resulting regression line
ax.plot(X, y_pred, color='#e74c3c', linewidth=3, label='OLS Fit Prediction')

# Formats
ax.set_xlabel("Predictor (X Variable)", fontsize=12, fontweight='bold', color='#333')
ax.set_ylabel("Response (Y Variable)", fontsize=12, fontweight='bold', color='#333')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

# Pass figure to streamlit
st.pyplot(fig)

with st.expander("Show the Raw Data Table"):
    st.dataframe(pd.DataFrame({'Predictor (X)': X.flatten(), 'Target (Y)': y}), use_container_width=True)

st.sidebar.divider()
st.sidebar.info("Developed for quick Linear Regression demonstrations visually.")
