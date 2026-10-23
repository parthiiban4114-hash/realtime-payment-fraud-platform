
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_transaction_prediction():
    payload = {
        "amount": 181,
        "step": 356,
        "oldbalanceOrg": 181,
        "oldbalanceDest": 0,
        "type": "TRANSFER",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["fraud_probability"] <= 1
    assert result["threshold"] == 0.20
    assert result["risk_status"] in [
        "FLAG_FOR_REVIEW",
        "NOT_FLAGGED",
    ]
    assert "disclaimer" in result


def test_negative_amount_is_rejected():
    payload = {
        "amount": -1,
        "step": 356,
        "oldbalanceOrg": 181,
        "oldbalanceDest": 0,
        "type": "TRANSFER",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_transaction_type_is_rejected():
    payload = {
        "amount": 500,
        "step": 356,
        "oldbalanceOrg": 2000,
        "oldbalanceDest": 5000,
        "type": "UNKNOWN",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
