import streamlit as st
import pickle
import numpy as np
from xgboost import XGBClassifier

# Loading the XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    xgboost_model = pickle.load(file)

st.title("Propensity Modeling App")
 # Sidebar with input fields
st.sidebar.header("Input Features")

no_of_days_active = st.sidebar.slider("No. of Days Active", 0.0, 1.0, 0.5)
R = st.sidebar.slider("R", 0.0, 1.0, 0.5)
F = st.sidebar.slider("F", 0.0, 1.0, 0.5)
M = st.sidebar.slider("M", 0.0, 1.0, 0.5)

avg_time_between_purchase = st.sidebar.slider("Avg. Time Between Purchase", 0.0, 1.0, 0.5)
loyalty_levels = {
        "Bronze": st.sidebar.checkbox("Bronze"),
        "Silver": st.sidebar.checkbox("Silver"),
        "Gold": st.sidebar.checkbox("Gold"),
        "Platinum": st.sidebar.checkbox("Platinum")
    }

input_data = np.array([no_of_days_active, R, F, M, avg_time_between_purchase] + list(loyalty_levels.values())).reshape(1, -1)

if st.button("Predict"):
        
        prediction = xgboost_model.predict(input_data)[0]
        prediction_proba = xgboost_model.predict_proba(input_data)[:, 1][0]

       
        st.success(f"Predicted Class: {prediction}")
        st.info(f"Prediction Probability: {prediction_proba:.4f}")
