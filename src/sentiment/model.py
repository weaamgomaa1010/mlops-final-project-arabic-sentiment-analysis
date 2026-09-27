from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from .preprocessing import tokenize_text


class SentimentModel:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )

        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = tokenize_text(text)

        inputs = {
            key: torch.tensor([value])
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_id = torch.argmax(outputs.logits, dim=-1).item()

        label = self.model.config.id2label[predicted_id]

        return {
            "label": label,
            "label_id": predicted_id
        }