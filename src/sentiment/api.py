from fastapi import FastAPI
from pydantic import BaseModel
from .predict import predict
from .config import MODEL_NAME, MAX_LENGTH
from .logging_config import setup_logging, correlation_id_ctx
import logging
import uuid

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Arabic Sentiment Analysis API",
    version="0.1.0"
)

@app.middleware("http")
async def add_correlation_id(request, call_next):
    correlation_id = request.headers.get(
        "X-Correlation-ID",
        str(uuid.uuid4())
    )

    token = correlation_id_ctx.set(correlation_id)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Request started")

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id

        logger.info("Request completed")

        return response
    except Exception:
        logger.exception("Request failed")
        raise
    finally:
        correlation_id_ctx.reset(token)


setup_logging()


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "model_name":MODEL_NAME,
        "max_length": MAX_LENGTH
    }


@app.post("/predict")
def predict_sentiment(request: PredictionRequest):
    return predict(request.text)


@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]):
    return [predict(request.text) for request in requests]
