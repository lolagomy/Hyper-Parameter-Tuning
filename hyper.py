import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load("tuned_decision_tree.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Define the Streamlit app
st.title("📊 Expresso Churn Prediction App")
st.write("Predict whether a customer will churn or not.")

# Create user input fields
customer_data = {}

# Sample categorical feature (modify as per dataset)
categorical_features = ['TENURE']
for feature in categorical_features:
    options = label_encoders[feature].classes_
    customer_data[feature] = st.selectbox(f"Select {feature}", options)

# Sample numerical features (modify as per dataset)
numerical_features = ['TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'MRG', 'REGULARITY']
for feature in numerical_features:
    customer_data[feature] = st.number_input(f"Enter {feature}", min_value=0.0, step=0.1)


# Preprocess user input
input_df = pd.DataFrame([customer_data])

# Encode categorical values
for feature in categorical_features:
    input_df[feature] = label_encoders[feature].transform(input_df[feature])

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    result = "✅ Customer will NOT churn" if prediction[0] == 0 else "⚠️ Customer WILL churn"
    st.subheader(result)
