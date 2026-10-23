
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Payment Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Real-Time Payment Fraud Detection")
st.caption("Machine learning-powered transaction risk assessment")

# Check whether the backend API is running
try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=3,
    )
    health_response.raise_for_status()
    st.success("Fraud detection API is connected.")
except requests.RequestException:
    st.error(
        "Cannot connect to the API. "
        "Start FastAPI in a separate terminal first."
    )
    st.code("python -m uvicorn api.main:app --reload")
    st.stop()

st.subheader("Transaction details")

with st.form("fraud_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=181.0,
            step=10.0,
        )

        transaction_type = st.selectbox(
            "Transaction type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
        )

        step = st.number_input(
            "Simulation time step",
            min_value=1,
            value=356,
            step=1,
        )

    with col2:
        oldbalance_org = st.number_input(
            "Origin account balance before transaction",
            min_value=0.0,
            value=181.0,
            step=10.0,
        )

        oldbalance_dest = st.number_input(
            "Destination account balance before transaction",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
    )

if submitted:
    payload = {
        "amount": amount,
        "step": int(step),
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest,
        "type": transaction_type,
    }

    try:
        with st.spinner("Analyzing transaction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

        if response.status_code == 200:
            result = response.json()

            probability = result["fraud_probability"]
            threshold = result["threshold"]

            st.subheader("Prediction results")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Fraud probability",
                f"{probability:.2%}",
            )
            metric2.metric(
                "Review threshold",
                f"{threshold:.0%}",
            )
            metric3.metric(
                "Risk status",
                result["risk_status"],
            )

            st.progress(min(max(probability, 0.0), 1.0))

            if result["risk_status"] == "FLAG_FOR_REVIEW":
                st.error("⚠️ Transaction flagged for investigation.")
            else:
                st.success("Transaction was not flagged by the model.")

            st.info(result["message"])
            st.caption(result["disclaimer"])

            with st.expander("View submitted transaction"):
                st.json(payload)

            with st.expander("View complete API response"):
                st.json(result)

        else:
            st.error(f"API returned HTTP {response.status_code}.")
            st.code(response.text)

    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")

st.divider()
st.caption(
    "Educational prototype using synthetic PaySim data. "
    "A model score is not proof of fraud or a production authorization decision."
)
