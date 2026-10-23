
from pathlib import Path
from typing import Literal
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from api.logging_config import logger


ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "temporal_random_forest.joblib"

FRAUD_THRESHOLD = 0.20
MODEL_NAME = "temporal_random_forest"


# In-memory monitoring metrics.
# These counters reset whenever the API process restarts.
prediction_metrics = {
    "total_predictions": 0,
    "flagged_transactions": 0,
    "not_flagged_transactions": 0,
    "prediction_errors": 0,
    "total_prediction_time_ms": 0.0,
}


app = FastAPI(
    title="Real-Time Payment Fraud Detection API",
    description="Scores payment transactions using a trained ML model.",
    version="1.0.0",
)


# Log incoming requests and their outcomes.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "event=request_completed request_id=%s "
            "method=%s path=%s status_code=%s "
            "duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "event=request_failed request_id=%s "
            "method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


# Load the trained model once when the API starts.
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}. "
        "Run python -m src.validate_temporal first."
    )

model = joblib.load(MODEL_FILE)


class TransactionRequest(BaseModel):
    amount: float = Field(ge=0)
    step: int = Field(ge=1)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    type: Literal[
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]


@app.get("/")
def home():
    return {
        "service": "Payment Fraud Detection API",
        "status": "running",
        "model": MODEL_NAME,
        "experimental_threshold": FRAUD_THRESHOLD,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
def get_metrics():
    total = prediction_metrics["total_predictions"]

    average_prediction_time_ms = (
        prediction_metrics["total_prediction_time_ms"] / total
        if total > 0
        else 0.0
    )

    return {
        "model": MODEL_NAME,
        "threshold": FRAUD_THRESHOLD,
        "total_predictions": total,
        "flagged_transactions": prediction_metrics[
            "flagged_transactions"
        ],
        "not_flagged_transactions": prediction_metrics[
            "not_flagged_transactions"
        ],
        "prediction_errors": prediction_metrics[
            "prediction_errors"
        ],
        "average_prediction_time_ms": round(
            average_prediction_time_ms, 2
        ),
    }


@app.post("/predict")
def predict_fraud(
    transaction: TransactionRequest,
    request: Request,
):
    request_id = getattr(
        request.state, "request_id", "unknown"
    )

    try:
        prediction_start = time.perf_counter()

        input_data = pd.DataFrame(
            [transaction.model_dump()]
        )

        fraud_probability = float(
            model.predict_proba(input_data)[0][1]
        )

        prediction_duration_ms = (
            time.perf_counter() - prediction_start
        ) * 1000

        flagged = fraud_probability >= FRAUD_THRESHOLD

        # Update monitoring metrics after successful inference.
        prediction_metrics["total_predictions"] += 1
        prediction_metrics["total_prediction_time_ms"] += (
            prediction_duration_ms
        )

        if flagged:
            prediction_metrics["flagged_transactions"] += 1
            risk_status = "FLAG_FOR_REVIEW"
        else:
            prediction_metrics["not_flagged_transactions"] += 1
            risk_status = "NOT_FLAGGED"

        logger.info(
            "event=prediction_completed request_id=%s "
            "model=%s risk_status=%s threshold=%.2f "
            "prediction_duration_ms=%.2f",
            request_id,
            MODEL_NAME,
            risk_status,
            FRAUD_THRESHOLD,
            prediction_duration_ms,
        )

        return {
            "fraud_probability": round(
                fraud_probability, 6
            ),
            "threshold": FRAUD_THRESHOLD,
            "risk_status": risk_status,
            "model": MODEL_NAME,
            "message": (
                "Transaction flagged for investigation."
                if flagged
                else "Transaction was not flagged by the model."
            ),
            "disclaimer": (
                "Experimental model output; not a confirmed "
                "fraud finding or an automatic authorization decision."
            ),
        }

    except Exception as exc:
        prediction_metrics["prediction_errors"] += 1

        logger.exception(
            "event=prediction_failed request_id=%s model=%s",
            request_id,
            MODEL_NAME,
        )

        raise HTTPException(
            status_code=500,
            detail="Fraud prediction failed.",
        ) from exc
