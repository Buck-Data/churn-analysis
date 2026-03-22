import streamlit as st
import pandas as pd
import pickle

with open('models/logistic_regression_model.pkl', 'rb') as f:
    modell = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Churn Prediction", page_icon="📊")

st.title("Customer Churn Prediction")
st.write("Nur die Faktoren die wirklich einen Einfluss auf Churn haben.")

# Info Box
st.info("Basierend auf unserer Analyse haben diese 7 Faktoren den "
        "stärksten Einfluss auf Kundenkündigung.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Vertrag & Kosten")
    contract = st.selectbox(
        "Vertragsart",
        ["Month-to-month", "One year", "Two year"],
        help="Stärkster einzelner Churn-Faktor!"
    )
    tenure = st.slider(
        "Vertragslaufzeit (Monate)", 0, 72, 12,
        help="Neue Kunden kündigen am häufigsten"
    )
    monthly = st.slider(
        "Monatliche Kosten (USD)", 18, 120, 65,
        help="Churner zahlen im Median 15 USD mehr"
    )
    payment = st.selectbox(
        "Zahlungsmethode",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        help="Electronic Check Kunden churnen 3x häufiger"
    )

with col2:
    st.subheader("Services")
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        help="Fiber optic Kunden haben höchste Churn Rate"
    )
    security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"],
        help="Reduziert Churn von 41% auf 14%"
    )
    tech = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"],
        help="Reduziert Churn von 41% auf 15%"
    )

st.divider()

if st.button("Churn Risiko berechnen", type="primary", use_container_width=True):

    # Unwichtige Features – Standardwerte gesetzt
    input_data = pd.DataFrame([{
        # Standardwerte für schwache Features
        'gender'          : 1,
        'SeniorCitizen'   : 0,
        'Partner'         : 1,
        'Dependents'      : 0,
        'PhoneService'    : 1,
        'MultipleLines'   : 0,
        'OnlineBackup'    : 0,
        'DeviceProtection': 0,
        'StreamingTV'     : 0,
        'StreamingMovies' : 0,
        'PaperlessBilling': 1,

        'tenure'          : tenure,
        'MonthlyCharges'  : monthly,
        'TotalCharges'    : monthly * tenure,
        'Contract'        : {"Month-to-month": 0,
                             "One year": 1,
                             "Two year": 2}[contract],
        'PaymentMethod'   : {"Bank transfer (automatic)": 0,
                             "Credit card (automatic)": 1,
                             "Electronic check": 2,
                             "Mailed check": 3}[payment],
        'InternetService' : {"DSL": 0, "Fiber optic": 1, "No": 2}[internet],
        'OnlineSecurity'  : {"No": 0, "No internet service": 1, "Yes": 2}[security],
        'TechSupport'     : {"No": 0, "No internet service": 1, "Yes": 2}[tech],
    }])

    input_scaled = scaler.transform(input_data)
    churn_prob = modell.predict_proba(input_scaled)[0][1]
    churn_pred = modell.predict(input_scaled)[0]

  
    if churn_pred == 1:
        st.error(f"Hohes Churn Risiko: {churn_prob*100:.1f}%")
        st.write("Dieser Kunde ist gefährdet zu kündigen.")
    else:
        st.success(f"Niedriges Churn Risiko: {churn_prob*100:.1f}%")
        st.write("Dieser Kunde ist wahrscheinlich loyal.")

    st.progress(float(churn_prob))
    
    # Erklärung der Risikofaktoren
    st.subheader("Warum dieses Ergebnis?")
    faktoren = []
    if contract == "Month-to-month":
        faktoren.append("Monatsvertrag erhöht Churn-Risiko stark")
    if payment == "Electronic check":
        faktoren.append("Electronic Check ist Hochrisiko-Zahlungsmethode")
    if tenure < 12:
        faktoren.append("Neukunde in der kritischen ersten Phase")
    if security == "No":
        faktoren.append("Kein Online Security Service")
    if tech == "No":
        faktoren.append("Kein Tech Support Service")
    if internet == "Fiber optic":
        faktoren.append("Fiber optic Kunden haben höchste Churn Rate")

    if faktoren:
        for f in faktoren:
            st.warning(f)