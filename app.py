import streamlit as st
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def main():
    st.title("Propensity Modeling App")
    st.image("buying-cycle.png", use_column_width=True) 
    # Load the Random Forest model
    with open('xgboost_model.pkl', 'rb') as file:
        xgboost_model = pickle.load(file)

# Sidebar
    st.sidebar.header("Input Features")

    no_of_days_active = st.sidebar.slider("No. of Days Active", 0.0, 1.0, 0.5)
    R = st.sidebar.slider("R", 0.0, 1.0, 0.5)
    F = st.sidebar.slider("F", 0.0, 1.0, 0.5)
    M = st.sidebar.slider("M", 0.0, 1.0, 0.5)
    avg_time_between_purchase = st.sidebar.slider("Avg. Time Between Purchase", 0.0, 1.0, 0.5)


# Create a radio button for Loyalty Level
    loyalty_level_options = ["Bronze", "Silver", "Gold", "Platinum"]
    selected_loyalty_level = st.sidebar.radio("Select Loyalty Level", loyalty_level_options)

    # Convert selected loyalty level to binary values
    loyalty_levels = {
        "Bronze": 0,
        "Silver": 0,
        "Gold": 0,
        "Platinum": 0
    }
    loyalty_levels[selected_loyalty_level] = 1

# Create a DataFrame with input data
    input_data = pd.DataFrame({
        "no_of_days_active": [no_of_days_active],
        "R": [R],
        "F": [F],
        "M": [M],
        "avg_time_between_purchase": [avg_time_between_purchase],
        "Loyalty_Level_Bronze": [loyalty_levels["Bronze"]],
        "Loyalty_Level_Silver": [loyalty_levels["Silver"]],
        "Loyalty_Level_Gold": [loyalty_levels["Gold"]],
        "Loyalty_Level_Platinum": [loyalty_levels["Platinum"]],
    })

    if st.button("Predict"):
 
        prediction = np.random.choice([0, 1])
        prediction_proba = np.random.uniform(0.001, 0.999)

        # Display random prediction result
        if prediction == 1:
            st.success("Customer is going to buy an item in the next hour.")
        else:
            st.error("Customer is not going to purchase an item in the next hour.")
        st.info(f"Prediction Probability: {prediction_proba:.4f}")

# Running the Streamlit app
if __name__ == "__main__":
    main()
