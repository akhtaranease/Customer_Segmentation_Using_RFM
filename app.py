import streamlit as st
import pickle
import numpy as np
import pandas as pd

def main():
    st.title("🧠 Customer Segmentation App")
    st.image("Buying-Cycle .png", use_container_width=True)

    # Load trained model
    with open('xgboost_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)

    st.sidebar.header("🔧 Input Customer Behavior (Scaled 0–1)")

    # Input sliders with helper tooltips
    no_of_days_active = st.sidebar.slider(
        "Days Active (0–1)", 0.0, 1.0, 0.5,
        help="0 = customer rarely active, 1 = highly active"
    )
    R = st.sidebar.slider(
        "Recency (0–1)", 0.0, 1.0, 0.5,
        help="0 = recent purchase, 1 = long time ago"
    )
    F = st.sidebar.slider(
        "Frequency (0–1)", 0.0, 1.0, 0.5,
        help="0 = rare buyer, 1 = frequent buyer"
    )
    M = st.sidebar.slider(
        "Monetary (0–1)", 0.0, 1.0, 0.5,
        help="0 = low spender, 1 = big spender"
    )
    avg_time_between_purchase = st.sidebar.slider(
        "Avg. Time Between Purchase (0–1)", 0.0, 1.0, 0.5,
        help="0 = buys frequently, 1 = buys very occasionally"
    )

    # Data input for model
    input_df = pd.DataFrame({
        "no_of_days_active": [no_of_days_active],
        "R": [R],
        "F": [F],
        "M": [M],
        "avg_time_between_purchase": [avg_time_between_purchase]
    })

    # Predict button
    if st.button("Predict"):
        prediction = xgb_model.predict(input_df)[0]

        loyalty_map = {0: "Bronze", 1: "Silver", 2: "Gold", 3: "Platinum"}
        recommendation_map = {
            "Bronze": "🔁 Re-engage with win-back offers or one-time discounts.",
            "Silver": "📩 Encourage repeat purchases with occasional offers.",
            "Gold": "🏆 Maintain loyalty with personalized rewards.",
            "Platinum": "💎 Reward them with exclusive VIP perks and early access."
        }

        loyalty_label = loyalty_map.get(prediction, "Unknown")
        recommendation = recommendation_map.get(loyalty_label, "No suggestion available.")

        st.markdown(f"### 🎯 Predicted Loyalty Level: **{loyalty_label}**")
        st.markdown(f"### 💡 Recommendation: {recommendation}")

if __name__ == "__main__":
    main()
