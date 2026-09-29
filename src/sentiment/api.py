from fastapi import FastAPI
from pydantic import BaseModel
from .predict import predict
from .config import MODEL_NAME, MAX_LENGTH

app = FastAPI(
    title="Arabic Sentiment Analysis API",
    version="0.1.0"
)

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
