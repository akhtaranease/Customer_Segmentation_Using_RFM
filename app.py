import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit App UI
st.title("📊 Customer Loyalty Prediction")
st.markdown("Enter RFM metrics to predict customer loyalty level.")

st.sidebar.header("Input Features")

# Sliders for input
no_of_days_active = st.sidebar.slider("Days Active (0–1)", 0.0, 1.0, 0.5)
R = st.sidebar.slider("Recency (0–1)", 0.0, 1.0, 0.5)
F = st.sidebar.slider("Frequency (0–1)", 0.0, 1.0, 0.5)
M = st.sidebar.slider("Monetary (0–1)", 0.0, 1.0, 0.5)
avg_time = st.sidebar.slider("Avg. Time Between Purchase (0–1)", 0.0, 1.0, 0.5)

# Make prediction
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "no_of_days_active": no_of_days_active,
        "R": R,
        "F": F,
        "M": M,
        "avg_time_between_purchase": avg_time
    }])

    prediction = model.predict(input_df)[0]
    label = "Likely to Purchase Again" if prediction == 1 else "Not Likely to Purchase Again"
    st.success(f"🔮 Prediction: **{label}**")
