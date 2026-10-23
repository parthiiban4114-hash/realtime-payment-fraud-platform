# Real-Time Payment Fraud Detection & Risk Intelligence Platform

## 1. Project Overview

The **Real-Time Payment Fraud Detection & Risk Intelligence Platform** is a machine-learning application designed to identify potentially fraudulent payment transactions and flag them for investigation.

The platform uses transaction details to estimate fraud probability, applies a configurable decision threshold, exposes predictions through a REST API, and provides an interactive Streamlit dashboard.

This project demonstrates an end-to-end machine-learning workflow, from data exploration and model development to API deployment, monitoring, logging, and automated testing.

**Dataset:** PaySim synthetic mobile-money transaction dataset. This is simulated data, not real Indian UPI transaction data.

## 2. Business Objectives

- Detect potentially fraudulent payment transactions.
- Estimate the probability that a transaction is fraudulent.
- Flag high-risk transactions for further investigation.
- Provide a REST API for real-time prediction requests.
- Display prediction results through a user-friendly dashboard.
- Monitor prediction volume, flagged transactions, errors, and response time.
- Validate API behavior using automated tests.

## 3. System Architecture

```text
PaySim Synthetic Dataset
          |
          v
Data Quality Checks and EDA
          |
          v
Feature Engineering
          |
          v
Temporal Train/Test Split
          |
          v
Random Forest Model Training
          |
          v
Model Evaluation and Threshold Analysis
          |
          v
Trained Model (.joblib)
          |
          v
FastAPI Prediction Service
          |
          +----------------------+
          |                      |
          v                      v
Streamlit Dashboard        Monitoring Metrics
          |                      |
          v                      v
Prediction Results         Logs and Statistics
```

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Model persistence | Joblib |
| REST API | FastAPI |
| API server | Uvicorn |
| Input validation | Pydantic |
| Dashboard | Streamlit |
| API communication | Requests |
| Testing | Pytest, HTTPX |
| Logging | Python logging |
| Dataset | PaySim synthetic transactions |

## 5. Project Structure

```text
realtime_payment_fraud_platform/
|
|-- api/
|   |-- __init__.py
|   |-- main.py
|   |-- logging_config.py
|
|-- dashboard/
|   |-- app.py
|
|-- data/
|   |-- raw/
|   |   |-- transactions.csv
|   |
|   |-- processed/
|       |-- transactions_features.csv
|       |-- data_quality_report.json
|
|-- models/
|   |-- logistic_regression.joblib
|   |-- random_forest.joblib
|   |-- temporal_random_forest.joblib
|
|-- notebooks/
|
|-- reports/
|
|-- src/
|   |-- feature_engineering.py
|   |-- inspect_model.py
|   |-- train_model.py
|   |-- validate_temporal.py
|
|-- tests/
|   |-- test_api.py
|   |-- test_api_extended.py
|
|-- logs/
|
|-- .venv/
|-- README.md
```

*The structure above reflects the main project files. Your local directory may contain additional files.*

## 6. Dataset and Data Quality

The project uses the PaySim synthetic mobile-money transaction dataset.

The dataset contains transaction attributes such as:

- `step`: Simulation time step.
- `type`: Transaction type.
- `amount`: Transaction amount.
- `oldbalanceOrg`: Origin account balance before the transaction.
- `newbalanceOrig`: Origin account balance after the transaction.
- `oldbalanceDest`: Destination account balance before the transaction.
- `newbalanceDest`: Destination account balance after the transaction.
- `isFraud`: Fraud label.
- `isFlaggedFraud`: Dataset-provided flag.

### Data quality results

| Metric | Result |
|---|---:|
| Total transactions | 6,362,620 |
| Total columns | 11 |
| Duplicate rows | 0 |
| Missing values | 0 |
| Fraudulent transactions | 8,213 |
| Non-fraudulent transactions | 6,354,407 |
| Fraud rate | Approximately 0.1291% |
| Rows removed during cleaning | 0 |

The dataset is highly imbalanced. Therefore, accuracy alone is not a suitable measure of model performance.

## 7. Feature Engineering

The project explores transaction attributes and balance inconsistencies.

Two engineered features were evaluated during the initial modeling stage:

- `errorBalanceOrig`: Difference between the origin account's expected and observed post-transaction balance.
- `errorBalanceDest`: Difference between the destination account's expected and observed post-transaction balance.

The initial model evaluation showed that balance-derived features were highly influential. However, these features can depend on post-transaction information and may introduce data leakage when used to make an authorization-time decision.

For the more conservative temporal model, the input features are restricted to:

- `amount`
- `step`
- `oldbalanceOrg`
- `oldbalanceDest`
- `type`

The customer and destination identifiers are excluded from the model inputs.

## 8. Model Development and Evaluation

The project evaluates Logistic Regression and Random Forest models.

### Initial model comparison

| Model | Average Precision (PR-AUC) |
|---|---:|
| Logistic Regression | Approximately 0.5810 |
| Initial Random Forest | Approximately 0.9976 |

The initial Random Forest result was exceptionally high and required further investigation for possible data leakage and validation issues.

### Temporal validation

A separate temporal experiment trains the Random Forest using earlier simulation steps and evaluates it on later steps.

| Metric | Result at threshold 0.50 |
|---|---:|
| Average Precision (PR-AUC) | 0.7843 |
| Precision | 0.9982 |
| Recall | 0.3993 |
| F1-score | 0.5704 |
| True positives | 1,697 |
| False positives | 3 |
| False negatives | 2,553 |

These results illustrate the trade-off between catching fraudulent transactions and minimizing false alarms.

### Threshold analysis

The API currently uses an experimental threshold of `0.20`.

In the temporal threshold analysis, this threshold produced approximately:

| Metric | Result |
|---|---:|
| Precision | 97.90% |
| Recall | 49.32% |
| F1-score | 0.6559 |
| True positives | 2,096 |
| False positives | 45 |
| False negatives | 2,154 |

**Important:** The threshold of 0.20 is exploratory, not production-approved. It was selected using the same temporal test results used for evaluation, so the reported metrics are not an independent estimate of performance at the selected threshold.

Before production use, select the threshold on a separate validation period and report final results on a later, untouched test period.

## 9. FastAPI Endpoints

The application exposes the following endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Health and model-loaded status |
| GET | `/metrics` | Prediction and latency counters |
| POST | `/predict` | Fraud probability and risk status |
| GET | `/docs` | Interactive Swagger API documentation |

### Example prediction request

```json
{
  "amount": 181.0,
  "step": 356,
  "oldbalanceOrg": 181.0,
  "oldbalanceDest": 0.0,
  "type": "TRANSFER"
}
```

### Example prediction response

```json
{
  "fraud_probability": 0.339922,
  "threshold": 0.2,
  "risk_status": "FLAG_FOR_REVIEW",
  "model": "temporal_random_forest",
  "message": "Transaction flagged for investigation.",
  "disclaimer": "Experimental model output; not a confirmed fraud finding or an automatic authorization decision."
}
```

The probability is a model score. It should not be interpreted as proof of fraud or as a calibrated real-world fraud probability without additional validation.

## 10. Streamlit Dashboard

The Streamlit dashboard provides an interactive interface for submitting sample transactions and viewing predictions.

The dashboard supports:

- Entering transaction amount.
- Selecting transaction type.
- Entering the simulation step.
- Entering origin and destination balances.
- Sending the transaction to the FastAPI endpoint.
- Displaying fraud probability and decision threshold.
- Showing the resulting risk status.
- Displaying the API response and disclaimer.
- Checking API health.

The dashboard communicates with the API at `http://127.0.0.1:8000`.

## 11. Logging and Monitoring

The API includes request logging and basic monitoring.

### Logging

Logs capture information such as:

- Request ID.
- HTTP method and endpoint.
- HTTP response status.
- Request duration.
- Model prediction outcome.
- Prediction execution time.
- Prediction failures and exceptions.

A request ID is returned through the `X-Request-ID` response header to support basic request tracing.

### Monitoring metrics

The `/metrics` endpoint reports:

- Total predictions.
- Flagged transactions.
- Transactions not flagged.
- Prediction errors.
- Average prediction execution time.

**Monitoring limitation:** Metrics are stored in process memory and reset when the API restarts. They are not designed for durable production monitoring or multi-worker aggregation.

## 12. Automated Testing

The project includes API tests using Pytest and FastAPI's test client.

The tests cover:

- Health endpoint behavior.
- Successful prediction requests.
- Fraud probability and response fields.
- Negative amount validation.
- Invalid transaction type validation.
- Missing required fields.
- Invalid simulation steps.
- Zero amount validation.
- Prediction latency measurement.
- Request ID response headers.

Run the tests from the project root:

```powershell
python -m pytest tests -v
```

The latest reported test run passed all 9 tests, with one dependency deprecation warning.

## 13. Running the Application

### Prerequisites

- Python installed.
- Project virtual environment configured.
- Required packages installed.
- Trained model available at `models/temporal_random_forest.joblib`.

Run the following commands from the project root.

### Terminal 1: Start the API

```powershell
python -m uvicorn api.main:app --reload
```

Open the API documentation in your browser:

`http://127.0.0.1:8000/docs`

### Terminal 2: Start the dashboard

```powershell
python -m streamlit run dashboard/app.py
```

Use the dashboard to submit a transaction and inspect the prediction response.

### Run automated tests

```powershell
python -m pytest tests -v
```

## 14. Known Limitations

- PaySim is synthetic data and does not represent the complete complexity of real payment fraud.
- Temporal evaluation is more conservative than the initial random split, but does not prove performance in a live payment environment.
- The threshold of 0.20 is exploratory and requires independent validation.
- The model's output is not necessarily a calibrated probability.
- In-memory metrics reset on restart.
- Current monitoring is basic and does not include a persistent metrics store or alerting system.
- The API flags transactions for review; it does not automatically block payments.
- Production deployment would require authentication, authorization, rate limiting, secure transport, secrets management, and operational monitoring.

## 15. Future Enhancements

- Add a separate validation period for threshold selection.
- Evaluate the final model on an untouched later test period.
- Add precision-recall curves and threshold visualizations.
- Monitor data drift and changes in fraud patterns.
- Add persistent metrics and dashboards.
- Introduce authentication and API rate limiting.
- Add model versioning and reproducible training pipelines.
- Evaluate probability calibration and business-specific fraud costs.
- Integrate a human investigation workflow.
- Containerize the application with Docker.
- Add CI/CD automation and deployment checks.
- Evaluate performance against realistic latency and throughput requirements.

## 16. Key Interview Talking Points

**Why is accuracy insufficient?**

Fraud is rare in this dataset. A model can achieve high accuracy by predicting almost every transaction as non-fraudulent. Precision, recall, F1-score, and PR-AUC provide more useful insight.

**Why use temporal validation?**

In a real payment system, a model generally learns from historical transactions and scores future ones. Training on earlier steps and testing on later steps better approximates this workflow.

**Why tune the threshold?**

The threshold controls the trade-off between missed fraud and false alarms. The appropriate operating point depends on fraud losses, investigation capacity, and customer impact.

**Why FastAPI?**

FastAPI exposes the trained model through a REST interface that other applications can call.

**Why Streamlit?**

Streamlit provides a simple user interface for entering transactions and interpreting API responses without building a separate frontend.

**What does monitoring provide?**

Monitoring tracks request volume, prediction outcomes, errors, and latency. Production systems require durable metrics, alerting, access controls, and operational dashboards.

## 17. Final Summary

This project demonstrates an end-to-end machine-learning application for payment fraud risk scoring. It combines data quality analysis, feature engineering, model comparison, temporal validation, threshold analysis, REST API development, an interactive dashboard, logging, monitoring, and automated tests.

The current implementation is a learning and demonstration project. Any real-world deployment would require independent model validation, security hardening, production monitoring, and approval of the fraud decision policy.

---
**Project:** Real-Time Payment Fraud Detection & Risk Intelligence Platform  
**Domain:** Payment fraud detection  
**Dataset:** PaySim synthetic mobile-money transactions  
**Model:** Temporal Random Forest  
**API:** FastAPI  
**Dashboard:** Streamlit  
**Testing:** Pytest