# Numerical analysis
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import altair as alt
from mpl_toolkits.mplot3d import Axes3D

#streamlit platform 
import streamlit as st
# Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def read_data(file):
    if file is not None:
        df = pd.read_csv(file)
        return df
    return None

st.subheader("Upload Data File (CSV or Excel)")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
data = read_data(uploaded_file)


st.header("Model Configuration")


slope = st.slider("Slope", -5.0, 5.0, 1.0)

intercept = st.slider("Intercept", -50.0, 50.0, 0.0)
model=LinearRegression()
if data is not None:
    X = data.iloc[:, 0].values.reshape(-1, 1)
    y = data.iloc[:, 1].values
else:
    np.random.seed(0)
    X = np.random.rand(100, 1)
    y = slope * X + intercept

model.fit(X, slope * X + intercept)

st.header("Data Visualization")
df = pd.DataFrame({'X': X.flatten(), 'y': model.predict(X).flatten()})
regression_plot = alt.Chart(df).mark_line(color='red').encode(
    x='X',
    y='y',
)

data_plot = alt.Chart(df).mark_circle(size=60).encode(
    x='X',
    y='y',
)

st.altair_chart(data_plot + regression_plot, use_container_width=True)

if st.checkbox("Show Table"):
    st.dataframe(df)

if st.checkbox("Show Regression Plot"):
    st.altair_chart(regression_plot, use_container_width=True)

if st.checkbox("Show Histogram"):
    st.write("Histogram of y:")
    fig, ax = plt.subplots()
    ax.hist(y, bins=20, color='skyblue')
    st.pyplot(fig)

if st.checkbox("Show 3D Plot"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, y, model.predict(X), c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_zlabel('Predicted y')
    ax.set_title('3D Scatter Plot')
    st.pyplot(fig)

if st.checkbox("Perform KFold"):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)  # Set the number of folds
    scores = []
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        scores.append((r2, mae, mse))

    scores_df = pd.DataFrame(scores, columns=['R-squared', 'MAE', 'MSE'])
    st.subheader("KFold Analysis")
    st.dataframe(scores_df)
st.header("Predictions")
st.subheader("Choose X Value")
x_input = st.number_input("Enter a value for X:", value=0.5, step=0.1)

# Increase and decrease buttons for X value
col1, col2, col3 = st.columns([1, 1, 1])
if col2.button("Increase X"):
    x_input += 0.1
if col2.button("Decrease X"):
    x_input -= 0.1

prediction = model.predict([[x_input]])[0][0]
st.write(f"For X = {x_input:.2f}, the predicted y is {prediction:.2f}")



