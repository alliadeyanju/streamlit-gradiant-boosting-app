
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the best model pipeline (includes scaler)
model = joblib.load('gradient_boosting_model.pkl')

st.title('💳 Credit Card Fraud Detector')
st.markdown('Enter transaction details to predict if the transaction is fraudulent.')

col1, col2 = st.columns(2)
with col1:
    amt = st.number_input('Transaction Amount', value=100.0)
    city_pop = st.number_input('City Population', value=5000)

st.markdown('### Location Coordinates')
lat = st.number_input('User Latitude', value=40.0)
long = st.number_input('User Longitude', value=-70.0)
merch_lat = st.number_input('Merchant Latitude', value=40.1)
merch_long = st.number_input('Merchant Longitude', value=-70.1)

if st.button('Analyze Transaction'):
    # Features must match the training set order
    features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    input_data = [[amt, lat, long, city_pop, merch_lat, merch_long]]
    input_df = pd.DataFrame(input_data, columns=features)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f'⚠️ High Risk! Fraud Probability: {probability:.2%}')
    else:
        st.success(f'✅ Legitimate Transaction. Fraud Probability: {probability:.2%}')
