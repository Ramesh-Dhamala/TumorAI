
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "TumorAI API is running!"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["total_features"] == 30


def test_get_features():
    response = client.get("/features")

    assert response.status_code == 200

    features = response.json()["features"]

    assert len(features) == 30
    assert "id" not in features
    assert "diagnosis" not in features


def test_valid_prediction():
    features_response = client.get("/features")
    feature_names = features_response.json()["features"]

    sample_features = {
        feature: 0.0 for feature in feature_names
    }

    response = client.post(
        "/predict",
        json={"features": sample_features}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["result"] in ["Benign", "Malignant"]
    assert 0 <= result["malignant_probability"] <= 100
    assert "disclaimer" in result


def test_prediction_with_missing_feature():
    features_response = client.get("/features")
    feature_names = features_response.json()["features"]

    sample_features = {
        feature: 0.0 for feature in feature_names
    }

    sample_features.pop(feature_names[0])

    response = client.post(
        "/predict",
        json={"features": sample_features}
    )

    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_prediction_with_unexpected_feature():
    features_response = client.get("/features")
    feature_names = features_response.json()["features"]

    sample_features = {
        feature: 0.0 for feature in feature_names
    }

    sample_features["unexpected_feature"] = 1.0

    response = client.post(
        "/predict",
        json={"features": sample_features}
    )

    assert response.status_code == 422
    assert "unexpected_features" in response.json()["detail"]
