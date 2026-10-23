
import time

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


VALID_PAYLOAD = {
    "amount": 181,
    "step": 356,
    "oldbalanceOrg": 181,
    "oldbalanceDest": 0,
    "type": "TRANSFER",
}


def test_zero_amount_is_accepted_by_validation():
    payload = {**VALID_PAYLOAD, "amount": 0}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_missing_required_field_is_rejected():
    payload = VALID_PAYLOAD.copy()
    payload.pop("amount")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_step_is_rejected():
    payload = {**VALID_PAYLOAD, "step": 0}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_response_time():
    # Warm up the model/API before measuring.
    client.post("/predict", json=VALID_PAYLOAD)

    start = time.perf_counter()
    response = client.post("/predict", json=VALID_PAYLOAD)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200

    print(f"\nPrediction request time: {elapsed_ms:.2f} ms")

    # Report the timing without assuming a performance target.
    assert elapsed_ms > 0

def test_request_id_header():
    response = client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0
