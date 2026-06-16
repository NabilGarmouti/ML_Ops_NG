"""Frontend Streamlit pour tester l'API de classification."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Cars Cross-Sell", layout="wide")
st.title("Cars Cross-Sell")

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, history_tab = st.tabs(["Prediction", "Historique"])

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    with st.form("predict_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=44)
        driving_license = st.selectbox("Driving_License", [0, 1], index=1)
        region_code = st.number_input("Region_Code", min_value=0.0, value=28.0)
        previously_insured = st.selectbox("Previously_Insured", [0, 1], index=0)
        vehicle_age = st.selectbox("Vehicle_Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
        vehicle_damage = st.selectbox("Vehicle_Damage", ["No", "Yes"], index=1)
        annual_premium = st.number_input("Annual_Premium", min_value=0.0, value=40454.0)
        sales_channel = st.number_input("Policy_Sales_Channel", min_value=0.0, value=26.0)
        vintage = st.number_input("Vintage", min_value=0, value=217)
        submitted = st.form_submit_button("Predire")

    if submitted:
        payload = {
            "Gender": gender,
            "Age": int(age),
            "Driving_License": int(driving_license),
            "Region_Code": float(region_code),
            "Previously_Insured": int(previously_insured),
            "Vehicle_Age": vehicle_age,
            "Vehicle_Damage": vehicle_damage,
            "Annual_Premium": float(annual_premium),
            "Policy_Sales_Channel": float(sales_channel),
            "Vintage": int(vintage),
        }

        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel a l'API impossible : {exc}")
        else:
            prediction = int(result["prediction"])
            probability = float(result["probability"])
            st.metric("Prediction", prediction)
            st.progress(min(max(probability, 0.0), 1.0))
            st.write(f"Probabilite de souscription : {probability:.4f}")

with history_tab:
    st.subheader("Historique des previsions")
    st.info("Aucun journal de previsions : ajoutez un endpoint /predictions a l'API (bonus).")

