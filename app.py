import streamlit as st
import pickle
import pandas as pd

def main():
    st.title("🧠 Customer Segmentation App")
    st.image("Buying-Cycle .png", use_container_width=True)

    # ——— Load your trained XGBoost model ———
    # (Make sure 'xgboost_model.pkl' is in the same folder as this script)
    with open("xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    # ——— Sidebar: Input sliders (scaled 0–1) ———
    st.sidebar.header("🔧 Input Customer Behavior (Scaled 0–1)")

    no_of_days_active = st.sidebar.slider(
        "Days Active", 0.0, 1.0, 0.5,
        help="0 = barely active, 1 = very active"
    )
    R = st.sidebar.slider(
        "Recency", 0.0, 1.0, 0.5,
        help="0 = bought very recently, 1 = hasn’t bought in a long time"
    )
    F = st.sidebar.slider(
        "Frequency", 0.0, 1.0, 0.5,
        help="0 = rarely buys, 1 = buys very often"
    )
    M = st.sidebar.slider(
        "Monetary", 0.0, 1.0, 0.5,
        help="0 = low spender, 1 = high spender"
    )
    avg_time_between_purchase = st.sidebar.slider(
        "Avg. Time Between Purchase", 0.0, 1.0, 0.5,
        help="0 = purchases frequently (short gaps), 1 = long gaps between purchases"
    )

    # ——— Build a DataFrame containing exactly the five numeric features your model expects ———
    input_df = pd.DataFrame({
        "no_of_days_active": [no_of_days_active],
        "R": [R],
        "F": [F],
        "M": [M],
        "avg_time_between_purchase": [avg_time_between_purchase]
    })

    # ——— Guarantee the column order exactly matches what your XGBoost model was trained on ———
    model_features = [
        "no_of_days_active",
        "R",
        "F",
        "M",
        "avg_time_between_purchase"
    ]
    input_df = input_df[model_features]

    # ——— When “Predict” is clicked, run the model and display a friendly result ———
    if st.button("Predict"):
        # Predict loyalty tier (0–3) using your XGBoost model
        pred_label = xgb_model.predict(input_df)[0]

        # Map numeric label (0,1,2,3) → human-friendly tier name
        loyalty_map = {
            0: "Bronze",
            1: "Silver",
            2: "Gold",
            3: "Platinum"
        }
        loyalty_name = loyalty_map.get(pred_label, "Unknown")

        # Provide a one-sentence recommendation for each tier
        tips = {
            "Bronze": "🔁 Consider win-back email campaigns or small discounts to re-engage this customer.",
            "Silver": "📩 Send targeted promotions and product recommendations to encourage repeat purchases.",
            "Gold":   "🏆 Offer occasional VIP deals or loyalty points to maintain their high engagement.",
            "Platinum":"💎 Give this customer early access to new products or exclusive VIP perks."
        }
        recommendation = tips.get(loyalty_name, "")

        st.markdown(f"### 🎯 Predicted Loyalty Level: **{loyalty_name}**")
        if recommendation:
            st.markdown(f"### 💡 Recommendation: {recommendation}")

if __name__ == "__main__":
    main()
