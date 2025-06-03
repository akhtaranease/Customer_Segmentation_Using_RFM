import streamlit as st
import pickle
import pandas as pd

# Paste this at the top of your file (adjust the path to your pickle if needed)
EXPECTED_COLUMNS = [
    "no_of_days_active",
    "R",
    "F",
    "M",
    "avg_time_between_purchase",
    "Loyalty_Level_Bronze",
    "Loyalty_Level_Silver",
    "Loyalty_Level_Gold",
    "Loyalty_Level_Platinum"
]

def main():
    st.title("Customer Segmentation App")

    # Load your trained model
    with open("xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    # Sidebar sliders (all floats between 0 and 1)
    no_of_days_active = st.sidebar.slider("Days Active (0–1)", 0.0, 1.0, 0.50)
    R = st.sidebar.slider("Recency (0–1)", 0.0, 1.0, 0.50)
    F = st.sidebar.slider("Frequency (0–1)", 0.0, 1.0, 0.50)
    M = st.sidebar.slider("Monetary (0–1)", 0.0, 1.0, 0.50)
    avg_time_between_purchase = st.sidebar.slider(
        "Avg Time Between Purchase (0–1)", 0.0, 1.0, 0.50
    )

    # One-hot encode loyalty level exactly as during training
    choice = st.sidebar.radio("Select Loyalty Level", ["Bronze", "Silver", "Gold", "Platinum"])
    loyalty_map = {
        "Loyalty_Level_Bronze":   0,
        "Loyalty_Level_Silver":   0,
        "Loyalty_Level_Gold":     0,
        "Loyalty_Level_Platinum": 0
    }
    loyalty_map[f"Loyalty_Level_{choice}"] = 1

    # Build a single-row DataFrame with the exact columns the model expects
    row = {
        "no_of_days_active":          no_of_days_active,
        "R":                          R,
        "F":                          F,
        "M":                          M,
        "avg_time_between_purchase":  avg_time_between_purchase,
        "Loyalty_Level_Bronze":       loyalty_map["Loyalty_Level_Bronze"],
        "Loyalty_Level_Silver":       loyalty_map["Loyalty_Level_Silver"],
        "Loyalty_Level_Gold":         loyalty_map["Loyalty_Level_Gold"],
        "Loyalty_Level_Platinum":     loyalty_map["Loyalty_Level_Platinum"]
    }
    input_df = pd.DataFrame([row], columns=EXPECTED_COLUMNS)

    # Predict on button click
    if st.button("Predict"):
        pred = xgb_model.predict(input_df)[0]
        proba = xgb_model.predict_proba(input_df)[0, 1]
        if pred == 1:
            st.success(f"Customer WILL purchase again (P = {proba:.3f})")
        else:
            st.error(f"Customer will NOT purchase again (P = {proba:.3f})")

if __name__ == "__main__":
    main()
