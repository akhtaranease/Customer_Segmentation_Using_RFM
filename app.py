import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# App Title and Description
st.title("📊 Customer Purchase Prediction App")

# ✅ Correct image usage (standalone line)
st.image("customer_segmentation_banner.png", use_container_width=True)


# Description below the image
st.markdown("""
This tool predicts whether a customer is **likely to make another purchase**  
based on RFM (Recency, Frequency, Monetary) and behavioral features.

Use the sliders to simulate different customer profiles.
""")


# 🧾 Input Features – Sidebar
st.sidebar.header("🛠 Input Features")

no_of_days_active = st.sidebar.slider("No. of Days Active", 0.0, 1.0, 0.5)
R = st.sidebar.slider("Recency (R)", 0.0, 1.0, 0.5)
F = st.sidebar.slider("Frequency (F)", 0.0, 1.0, 0.5)
M = st.sidebar.slider("Monetary (M)", 0.0, 1.0, 0.5)
avg_time = st.sidebar.slider("Avg Time Between Purchase", 0.0, 1.0, 0.5)

# 🎖️ Loyalty Level
loyalty = st.sidebar.radio("Select Loyalty Level", ["Bronze", "Silver", "Gold", "Platinum"])

# One-hot encoding loyalty
loyalty_dict = {
    "Loyalty_Level_Bronze": 0,
    "Loyalty_Level_Silver": 0,
    "Loyalty_Level_Gold": 0,
    "Loyalty_Level_Platinum": 0
}
loyalty_dict[f"Loyalty_Level_{loyalty}"] = 1

# 🔍 Build input row
input_df = pd.DataFrame([{
    "no_of_days_active": no_of_days_active,
    "R": R,
    "F": F,
    "M": M,
    "avg_time_between_purchase": avg_time,
    "Loyalty_Level_Bronze": loyalty_dict["Loyalty_Level_Bronze"],
    "Loyalty_Level_Silver": loyalty_dict["Loyalty_Level_Silver"],
    "Loyalty_Level_Gold": loyalty_dict["Loyalty_Level_Gold"],
    "Loyalty_Level_Platinum": loyalty_dict["Loyalty_Level_Platinum"]
}])

# 🚀 Make prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
        st.write("Model output:", pred)

    prob = model.predict_proba(input_df)[0][pred] * 100  # Confidence %
    label = "✅ Likely to Purchase Again" if pred == 1 else "❌ Not Likely to Purchase Again"
    st.si dont get it
    uccess(f"Prediction: **{label}**\n\nConfidence: **{prob:.1f}%**")
