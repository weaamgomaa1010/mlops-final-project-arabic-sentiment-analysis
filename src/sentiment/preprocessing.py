from arabert.preprocess import ArabertPreprocessor
from transformers import AutoTokenizer
from .config import MODEL_NAME, MAX_LENGTH


arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_text(text: str) -> str:
    return arabert_prep.preprocess(text)


def tokenize_text(text: str) -> dict:
    processed_text = preprocess_text(text)

    return tokenizer(
        processed_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )