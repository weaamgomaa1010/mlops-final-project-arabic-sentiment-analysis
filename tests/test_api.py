from fastapi.testclient import TestClient
from sentiment.api import app


client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status" : "healthy"}


def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200

    data = response.json()

    assert "model_name" in data
    assert "max_length" in data


def test_predict():
    response = client.post(
        "/predict",
        json={"text": "الفيلم جميل جدًا"}
    )
    assert response.status_code == 200

    data = response.json()
    assert "label" in data
    assert "label_id" in data


def test_predict_batch():
    response = client.post(
        "/predict/batch",
        json=[
            {"text": "الفيلم جميل جدًا"},
            {"text": "الخدمة سيئة"}
        ]
    )
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2
    assert all("label" in item for item in data)
    assert all("label_id" in item for item in data)