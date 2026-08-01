import pickle
import streamlit as st
import numpy as np

st.title("Disease Prediction System!!!")

@st.cache_resource
def load_model():
    with open("dp.pkl", "rb") as file:
        return pickle.load(file)

try:
    dp = load_model()
except FileNotFoundError:
    st.error("Error: Please upload your 'dp.pkl' file to this GitHub repository.")
    dp = None

age = st.number_input("Enter Age : ", min_value=0, max_value=120, value=25)
hrb = st.number_input("Enter Heart Rate bpm : ", min_value=30, max_value=220, value=75)
bt = st.number_input("Enter Body Temperature : ", min_value=30.0, max_value=45.0, value=37.0)
os = st.number_input("Enter Oxygen Saturation : ", min_value=50, max_value=100, value=98)

if st.button("Predict!!"):
    if dp is not None:
        input_features = np.array([[age, hrb, bt, os]])
        res = dp.predict(input_features)
        st.warning(f"Expected Disease : {res}")
