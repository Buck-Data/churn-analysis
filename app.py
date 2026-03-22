import streamlit as st
import pandas as pd
import pickle

with open('models/churn_model.pkl', 'rb') as f:
    modell = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Gib die Kundendaten ein um das Churn-Risiko vorherzusagen.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Kundenprofil")
    gender = st.selectbox("Geschlecht", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Vertragslaufzeit (Monate)", 0, 72, 12)

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multi_lines = st.selectbox("Multiple Lines", 
                               ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", 
                            ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", 
                            ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", 
                          ["No", "Yes", "No internet service"])
    device = st.selectbox("Device Protection", 
                          ["No", "Yes", "No internet service"])
    tech = st.selectbox("Tech Support", 
                        ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", 
                      ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", 
                          ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Vertrag & Zahlung")
    contract = st.selectbox("Vertragsart",
                            ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Zahlungsmethode",
                           ["Electronic check", "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)"])
    monthly = st.slider("Monatliche Kosten (USD)", 18, 120, 65)
    total = st.number_input("Gesamtkosten (USD)", 
                            min_value=0.0, value=float(monthly * tenure))

if st.button("Churn Risiko berechnen", type="primary"):

    # Encoding – genau wie LabelEncoder alphabetisch sortiert!
    input_data = pd.DataFrame([{
        'gender'          : 0 if gender == "Female" else 1,
        'SeniorCitizen'   : 0 if senior == "No" else 1,
        'Partner'         : 0 if partner == "No" else 1,
        'Dependents'      : 0 if dependents == "No" else 1,
        'tenure'          : tenure,
        'PhoneService'    : 0 if phone == "No" else 1,
        'MultipleLines'   : {"No": 0, "No phone service": 1, "Yes": 2}[multi_lines],
        'InternetService' : {"DSL": 0, "Fiber optic": 1, "No": 2}[internet],
        'OnlineSecurity'  : {"No": 0, "No internet service": 1, "Yes": 2}[security],
        'OnlineBackup'    : {"No": 0, "No internet service": 1, "Yes": 2}[backup],
        'DeviceProtection': {"No": 0, "No internet service": 1, "Yes": 2}[device],
        'TechSupport'     : {"No": 0, "No internet service": 1, "Yes": 2}[tech],
        'StreamingTV'     : {"No": 0, "No internet service": 1, "Yes": 2}[tv],
        'StreamingMovies' : {"No": 0, "No internet service": 1, "Yes": 2}[movies],
        'Contract'        : {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        'PaperlessBilling': 0 if paperless == "No" else 1,
        'PaymentMethod'   : {"Bank transfer (automatic)": 0,
                             "Credit card (automatic)": 1,
                             "Electronic check": 2,
                             "Mailed check": 3}[payment],
        'MonthlyCharges'  : monthly,
        'TotalCharges'    : total
    }])

    # Skalieren & Vorhersage
    input_scaled = scaler.transform(input_data)
    churn_prob = modell.predict_proba(input_scaled)[0][1]
    churn_pred = modell.predict(input_scaled)[0]

    # Ergebnis anzeigen
    st.divider()

    if churn_pred == 1:
        st.error(f"Hohes Churn Risiko: {churn_prob*100:.1f}%")
        st.write("Dieser Kunde ist gefährdet zu kündigen.")
    else:
        st.success(f"Niedriges Churn Risiko: {churn_prob*100:.1f}%")
        st.write("Dieser Kunde ist wahrscheinlich loyal.")

    st.progress(float(churn_prob))
    st.caption(f"Churn Wahrscheinlichkeit: {churn_prob*100:.1f}%")