import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df = df.drop(columns=["Unnamed: 0", "company"])

    df = df.rename(
        columns={
            "review_description": "review",
            "rating": "label"
        }
    )

    df["label"] = df["label"].map({
        -1: "Negative",
        0: "Neutral",
        1: "Positive"
    })

    df = df.dropna(subset=["review", "label"])
    df = df.reset_index(drop=True)

    return df


def split_data(df: pd.DataFrame):
    X = df["review"]
    y = df["label"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1111,
        stratify=y_temp,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test