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
    payment_method = st.selectbox("Zahlungsmethode", options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    internet_service = st.selectbox("Internetdienst", options=['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online-Sicherheit", options=['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox("Technischer Support", options=['Yes', 'No'])   

if st.button("Vorhersage berechnen"):
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_mapping = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
    internet_mapping = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    security_mapping = {'Yes': 0, 'No': 1, 'No internet service': 2}
    support_mapping = {'Yes': 0, 'No': 1}

    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [tenure * MonthlyCharges],
        'Contract': [contract_mapping[contract]],
        'PaymentMethod': [payment_mapping[payment_method]],
        'InternetService': [internet_mapping[internet_service]],
        'OnlineSecurity': [security_mapping[online_security]],
        'TechSupport': [support_mapping[tech_support]]
    })

    input_scaled = scaler.transform(input_data)
    churn_prob = model.predict_proba(input_scaled)[0][1]
    churn_pred = model.predict(input_scaled)[0]

    st.divider()

    if churn_pred == 1:
        st.error(f"Der Kunde hat eine hohe Wahrscheinlichkeit von {churn_prob:.2%} abzuwandern.")
        st.write("Empfehlung: Kundenbindungsmaßnahmen ergreifen, z.B. Rabatte oder verbesserten Service anbieten.")
    else:
        st.success(f"Der Kunde hat eine niedrige Wahrscheinlichkeit von {churn_prob:.2%} abzuwandern.")
        st.write("Empfehlung: Weiterhin guten Service bieten und Kundenbindung stärken.")

    st.progress(float(churn_prob))
    st.caption(f"Churn Wahrscheinlichkeit: {churn_prob*100:.1f}%")
