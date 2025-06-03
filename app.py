import streamlit as st
import pickle
import pandas as pd

def main():
    st.title("🧠 Customer Segmentation App")
    st.image("Buying-Cycle .png", use_container_width=True)

    # Load your trained XGBoost model
    with open("xgboost_model.pkl", "rb") as file:
        xgb_model = pickle.load(file)

    # Sidebar sliders
    st.sidebar.header("🔧 Input Customer Behavior (Scaled 0–1)")

    no_of_days_active = st.sidebar.slider("Days Active (0–1)", 0.0, 1.0, 0.5)
    R = st.sidebar.slider("Recency (0–1)", 0.0, 1.0, 0.5)
    F = st.sidebar.slider("Frequency (0–1)", 0.0, 1.0, 0.5)
    M = st.sidebar.slider("Monetary (0–1)", 0.0, 1.0, 0.5)
    avg_time_between_purchase = st.sidebar.slider("Avg. Time Between Purchase (0–1)", 0.0, 1.0, 0.5)

    # Create input DataFrame
    input_data = pd.DataFrame({
        "no_of_days_active": [no_of_days_active],
        "R": [R],
        "F": [F],
        "M": [M],
        "avg_time_between_purchase": [avg_time_between_purchase]
    })

    # Make sure column order matches exactly what the model was trained on
    input_data = input_data[["no_of_days_active", "R", "F", "M", "avg_time_between_purchase"]]

    if st.button("Predict"):
        prediction = xgb_model.predict(input_data)[0]

        loyalty_map = {0: "Bronze", 1: "Silver", 2: "Gold", 3: "Platinum"}
        recommendation_map = {
            "Bronze": "🔁 Re-engage with win-back offers or one-time discounts.",
            "Silver": "📩 Encourage them with occasional offers and product updates.",
            "Gold": "🏆 Maintain loyalty with personalized rewards.",
            "Platinum": "💎 Reward them with VIP perks and early access."
        }

        loyalty_label = loyalty_map.get(prediction, "Unknown")
        recommendation = recommendation_map.get(loyalty_label, "No recommendation available.")

        st.markdown(f"### 🎯 Predicted Loyalty Level: **{loyalty_label}**")
        st.markdown(f"### 💡 Recommendation: {recommendation}")

if __name__ == "__main__":
    main()
