# 🧠 Customer Segmentation & Purchase Prediction App

![App Screenshot](Stream_Lit_App.png)

This project combines customer segmentation analysis with a predictive model to identify customers most likely to make another purchase.  
It demonstrates end-to-end skills in data preprocessing, feature engineering, model training, and interactive app deployment.

---

## 📊 Part 1: RFM Segmentation & Exploratory Analysis

We applied **Recency, Frequency, and Monetary (RFM)** analysis — where:  
- **Recency** = how recently a customer made a purchase  
- **Frequency** = how often they purchase  
- **Monetary** = how much they spend  
- These metrics helped us categorize customer loyalty levels and explore behavioral trends.

### ✅ Key Tasks:
- Cleaned and formatted customer transaction data  
- Generated RFM scores and assigned loyalty levels  
- Engineered new features: `no_of_days_active`, `avg_time_between_purchase`  
- Visualized trends using `pandas`, `matplotlib`, and `seaborn`

---

## 🤖 Part 2: Purchase Prediction Model

Built a **binary classification model** to predict whether a customer is *likely to purchase again*.

### ⚙️ ML Workflow:
- **Features**: `R`, `F`, `M`, `no_of_days_active`, `avg_time_between_purchase` + one-hot encoded loyalty level  
- **Target**: `Purchase (1)` or `No Purchase (0)`  
- **Class Balancing**: manually converted labels to binary; SMOTE considered  
- **Model**: `XGBoost Classifier`  
- **Tuning**: `RandomizedSearchCV` for hyperparameter optimization  
- **Evaluation**: classification report, confusion matrix, ROC AUC

---

## 🖥️ Part 3: Streamlit App Deployment

An interactive web app that lets users simulate customer profiles and view predictions in real time.

### 🎨 App Features:
- Sliders for RFM and behavior-based inputs  
- Radio buttons for loyalty tier (one-hot encoded automatically)  
- Live prediction with confidence percentage  
- Clean layout and professional banner image  
- Deployed via Streamlit Cloud

👉 **[Launch App](https://customersegmentationusingrfm-pyfhxyvkrcuiggshixtzec.streamlit.app/)**

---

## 🧠 Skills Demonstrated

- ✅ Data Cleaning & EDA  
- ✅ Feature Engineering  
- ✅ One-hot Encoding  
- ✅ Model Training (XGBoost)  
- ✅ Hyperparameter Tuning  
- ✅ Model Serialization (`.pkl`)  
- ✅ Streamlit App Development  
- ✅ GitHub Version Control  
- ✅ Streamlit Cloud Deployment

---

## 📁 Repository Contents

| File                          | Description                                   |
|-------------------------------|-----------------------------------------------|
| `Customer_Segmentation_RFM.ipynb` | EDA, feature engineering & model training |
| `app.py`                      | Streamlit app frontend/backend code           |
| `xgboost_model.pkl`           | Saved XGBoost model                           |
| `customer_segmentation_banner.png` | App visual header image                |
| `requirements.txt`            | Package dependencies for deployment           |

---

## 💼 Business Use Cases

- 🎯 Identify high-value or at-risk customers  
- 🔁 Test marketing scenarios with synthetic profiles  
- 📈 Support customer retention strategy  
- 💡 Turn behavioral data into actionable insights

---

## 🔮 Next Steps

- Add SHAP for model interpretability  
- Allow CSV file uploads for batch predictions  
- Automate loyalty level assignment

---

## 👤 Author

- [Anease Akhtar](https://github.com/akhtaranease)

---

## 🚀 Demo

Try the live app here:  
**https://customersegmentationusingrfm-pyfhxyvkrcuiggshixtzec.streamlit.app/**
