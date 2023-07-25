# Numerical analysis
import numpy as np
import pandas as pd
#visualization
import altair as alt
#streamlit platform 
import streamlit as st
# Model
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

alt.themes.enable('dark')
st.markdown("""
<style>
body {
    color: #333;
    font-family: 'Helvetica', 'Arial', sans-serif;
}

header {
    background-color: #333;
    padding: 20px;
    text-align: center;
}

h1, h2 {
    color: #FF6600;
}

.container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}

.stButton button {
    background-color: #FF6600;
    color: #fff;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
}

.stButton button:hover {
    background-color: #E64A00;
}

.stCheckbox label {
    font-size: 18px;
    color: #666;
}

.stHeader {
    background-color: #FF6600;
    color: #fff;
    padding: 10px;
    text-align: center;
    border-radius: 5px;
    margin-top: 20px;
}

.stDataFrame {
    background-color: #fff;
    color: #333;
}

.stAltairChart {
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

def generate_data_linear(seed, slope, intercept, num_data_points):
    rng = np.random.RandomState(seed)
    x = 10 * rng.rand(num_data_points) + 0.3 * rng.randn(num_data_points)  # Adding noise to 'x'
    y = slope * x + intercept + rng.randn(num_data_points)
    return x, y

def generate_data_sine(seed, slope, intercept, num_data_points):
    rng = np.random.RandomState(seed)
    x = 10 * rng.rand(num_data_points) + rng.randn(num_data_points)  # Adding noise to 'x'
    y = np.sin(x) + 0.3 * rng.randn(num_data_points)
    return x, y

def generate_data_exponential(seed, slope, intercept, num_data_points):
    rng = np.random.RandomState(seed)
    x = 10 * rng.rand(num_data_points) + rng.randn(num_data_points)  # Adding noise to 'x'
    y = np.exp(x) + 0.3 * rng.randn(num_data_points)
    return x, y

def generate_data_polynomial(seed, slope, intercept, num_data_points):
    rng = np.random.RandomState(seed)
    x = 10 * rng.rand(num_data_points) + rng.randn(num_data_points)  # Adding noise to 'x'
    y = slope * x**2 + intercept + rng.randn(num_data_points)
    return x, y

def generate_data_gaussian_process(seed, slope, intercept, num_data_points):
    rng = np.random.RandomState(seed)
    x = 10 * rng.rand(num_data_points)
    y = slope * x + intercept + rng.randn(num_data_points)  # Adding noise to 'y'
    
    # Use Gaussian Process Regressor to generate data
    kernel = C() * RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=seed)
    gp.fit(x[:, np.newaxis], y)
    yfit = gp.predict(np.linspace(0, 10, num_data_points)[:, np.newaxis])
    return x,y

# Slider Section
seed = st.slider("Random seed", 1, 50, 10)
slope = st.slider("Slope", -10.0, 10.0, 2.0, 0.1)
intercept = st.slider("Intercept", -10.0, 10.0, -5.0, 0.1)
num_data_points = st.slider("Number of Data Points", 10, 5000, 50)

# Checkbox Section for selecting data generation functions
st.markdown("### Data Generation Functions")
generate_linear = st.checkbox("Linear Function")
generate_sine = st.checkbox("Sine Function")
generate_exponential = st.checkbox("Exponential Function")
generate_polynomial = st.checkbox("Polynomial Function")
generate_gaussian_process = st.checkbox("Gaussian Process Function")

# Generate data based on the selected functions
x, y = [], []

if generate_linear:
    x_temp, y_temp = generate_data_linear(seed, slope, intercept, num_data_points)
    x.extend(x_temp)
    y.extend(y_temp)

if generate_sine:
    x_temp, y_temp = generate_data_sine(seed, slope, intercept, num_data_points)
    x.extend(x_temp)
    y.extend(y_temp)

if generate_exponential:
    x_temp, y_temp = generate_data_exponential(seed, slope, intercept, num_data_points)
    x.extend(x_temp)
    y.extend(y_temp)

if generate_polynomial:
    x_temp, y_temp = generate_data_polynomial(seed, slope, intercept, num_data_points)
    x.extend(x_temp)
    y.extend(y_temp)

if generate_gaussian_process:
    x_temp, y_temp = generate_data_gaussian_process(seed, slope, intercept, num_data_points)
    x.extend(x_temp)
    y.extend(y_temp)

if len(x) > 0:
    # Create DataFrame for the scatter plot
    data = pd.DataFrame({'x': x, 'y': y})

    # Create linear regression model
    model = LinearRegression(fit_intercept=True)
    model.fit(np.array(x)[:, np.newaxis], y)

    # Generate data points for regression line
    xfit = np.linspace(0, 10, 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    # Scatter plot using Altair
    scatter_plot = alt.Chart(data).mark_circle(size=60, color='steelblue').encode(
        x='x',
        y='y',
        tooltip=['x', 'y']
    ).interactive()

    # Line chart for the regression line using Altair
    regression_line_chart = alt.Chart(pd.DataFrame({'x': xfit, 'y': yfit})).mark_line(
        color='orange', strokeWidth=2
    ).encode(
        x='x',
        y='y'
    )

    # Combine the scatter plot and regression line chart using Altair
    combined_chart = (scatter_plot + regression_line_chart).properties(
        width=600,
        height=400
    )

    # Show the combined chart
    st.altair_chart(combined_chart)
else:
    st.write("Please select at least one data generation function.")