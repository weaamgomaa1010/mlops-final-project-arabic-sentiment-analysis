from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "CompanyReviews.csv"

MODEL_PATH = BASE_DIR / "models" / "arabert-sentiment"

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

MAX_LENGTH = 128

LABEL2ID = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}