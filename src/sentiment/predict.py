from .model import SentimentModel
from .config import MODEL_PATH

model = SentimentModel(MODEL_PATH)

def predict(text: str) -> dict:
    return model.predict(text)