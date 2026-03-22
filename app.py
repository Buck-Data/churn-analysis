import pickle
import streamlit as st
import pandas as pd
import xgboost as xgb

with open('models/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Kundenabwanderungsanalyse")
st.write("Geben Sie die Kundendaten ein, um die Wahrscheinlichkeit der Abwanderung  zu berechnen.")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Vertragslaufzeit (Monate)", min_value=0, max_value=72, value=12)
    MonthlyCharges = st.number_input("Monatliche Kosten", min_value=0.0, value=70.0)
    contract = st.selectbox("Vertragsart", options=['Month-to-month', 'One year', 'Two year'])

with col2:
    paymenrt_method = st.selectbox("Zahlungsmethode", options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    internet_service = st.selectbox("Internetdienst", options=['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online-Sicherheit", options=['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox("Technischer Support", options=['Yes', 'No'])   

if st.button("Vorhersage berechnen"):
    st.write("Berechnung läuft...")