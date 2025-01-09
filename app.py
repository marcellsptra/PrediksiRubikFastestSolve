import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('model/model.pkl')

# Load dataset
data = pd.read_csv('model/solves.csv')

# Title
st.title("Prediksi Fastest Solve Rubik")

# Sidebar input
st.sidebar.header("Input Waktu Solve")
solve_1 = st.sidebar.number_input("Solve 1", min_value=0, step=1)
solve_2 = st.sidebar.number_input("Solve 2", min_value=0, step=1)
solve_3 = st.sidebar.number_input("Solve 3", min_value=0, step=1)
solve_4 = st.sidebar.number_input("Solve 4", min_value=0, step=1)
solve_5 = st.sidebar.number_input("Solve 5", min_value=0, step=1)

# Predict button
if st.sidebar.button("Prediksi"):
    input_data = np.array([[solve_1, solve_2, solve_3, solve_4, solve_5]])
    prediction = model.predict(input_data)
    st.subheader("Hasil Prediksi")
    st.write(f"Fastest Solve: {prediction[0]:.2f} detik")

# Display dataset
st.subheader("Dataset")
st.write(data.head())
