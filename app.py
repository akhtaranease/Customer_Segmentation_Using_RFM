import streamlit as st
import pickle
import numpy as np
import pandas as pd

def main():
    st.title("🧠 Customer Segmentation App")
    st.image("Buying-Cycle .png", use_container_width=True)

    # Load your trained model
    with open('xgboost_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)

    st.sidebar.header("🔧 Input Customer Behavior (Scaled 0–1)")

    # Sliders with helper text
    no_of_days_active = st.sidebar.slider(
        "Days Active", 0.0, 1.0, 0.5,
        help="0 = low activity, 1 = highly active"
    )
    R = st.sidebar.slider(
        "Recency", 0.0, 1.0, 0.5,
        help="0 = recently purchased, 1 = long time ago"
    )
    F = st.sidebar.slider(
        "Frequency", 0.0, 1.0, 0.5,
        help="0 = rarely buys, 1 = buys often"
    )
    M = st.sidebar.slider(
        "Monetary", 0.0, 1.0, 0.5,
        help="0 = low spender, 1 = high spender"
    )
    avg_time_between_purchase = st.sidebar.slider(
        "Avg. Time Between Purchase", 0.0, 1.0, 0.5,
        help="0 = buys frequently, 1 = long gaps between buys"
    )

    # Create input DataFrame
    input_df = pd.DataFrame({
        "no_of_days_active": [no_of_days_active],
        "R": [R],
        "F": [F],
        "M": [M],
        "avg_time_between_purchase": [avg_time_between_purchase]
    })

    # Ensure feature order matches the trained model
    model_features = ['no_of_days_active', 'R', 'F', 'M', 'avg_time_between_purchase']
    input_df = input_df[model_features]

    if st.button("Predict"):
        prediction = xgb_model.predict(input_df)[0]

        # Map predicted label to loyalty tier
        loyalty_map = {0: "Bronze", 1: "Silver", 2: "Gold", 3: "Platinum"}
        tips = {
            "Bronze": "🔁 Re-engage with win-back offers or small discounts.",
            "Silver": "📩 Encourage regular purchases with special emails.",
            "Gold": "🏆 Keep them loyal with occasional exclusive deals.",
            "Platinum": "💎 Give VIP perks, early access, or loyalty bonuses."
        }

        loyalty_label = loyalty_map.get(prediction, "Unknown")
        st.markdown(f"### 🎯 Predicted Loyalty Level: **{loyalty_label}**")
        st.markdown(f"### 💡 Recommendation: {tips.get(loyalty_label, '')}")

if __name__ == "__main__":
    main()
