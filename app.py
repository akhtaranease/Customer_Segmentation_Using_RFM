import streamlit as st
import pickle
import pandas as pd

# Load trained XGBoost model
with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# Set up page
st.title("📊 Customer Loyalty Prediction")
st.write("Enter customer RFM data to predict loyalty segment.")

# Sidebar for inputs
st.sidebar.header("Input RFM Features")

no_of_days_active = st.sidebar.slider("No. of Days Active (0–1)", 0.0, 1.0, 0.5)
R = st.sidebar.slider("Recency (0–1)", 0.0, 1.0, 0.5)
F = st.sidebar.slider("Frequency (0–1)", 0.0, 1.0, 0.5)
M = st.sidebar.slider("Monetary Value (0–1)", 0.0, 1.0, 0.5)
avg_time_between_purchase = st.sidebar.slider("Avg Time Between Purchase (0–1)", 0.0, 1.0, 0.5)

# Build input DataFrame with correct feature names
input_data = pd.DataFrame([{
    "no_of_days_active": no_of_days_active,
    "R": R,
    "F": F,
    "M": M,
    "avg_time_between_purchase": avg_time_between_purchase
}])

# Prediction logic
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    loyalty_labels = {
        0: "Bronze",
        1: "Silver",
        2: "Gold",
        3: "Platinum"
    }

    recs = {
        "Bronze": "🔁 Send win-back offers or discounts to re-engage.",
        "Silver": "📧 Keep engaged with regular updates and perks.",
        "Gold": "🏆 Reward loyalty with tailored promotions.",
        "Platinum": "💎 VIP treatment — exclusive deals and early access."
    }

    label = loyalty_labels.get(prediction, "Unknown")
    st.success(f"🎯 Predicted Loyalty Level: **{label}**")
    st.info(recs.get(label, "No recommendation available."))
