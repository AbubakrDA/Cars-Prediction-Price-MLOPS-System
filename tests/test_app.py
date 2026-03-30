from fastapi.testclient import TestClient
from DeployfastApi import app
import os
import joblib

client = TestClient(app)

def test_index():
    """Verify that the API root is accessible."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to Car Price Prediction API"

def test_model_exists():
    """Ensure the model artifact is present in the root directory."""
    assert os.path.exists("car_price_prediction_model.pkl")

def test_prediction():
    """Verify that the prediction endpoint returns a valid float value."""
    payload = {
        "Present_Price": 5.59,
        "Kms_Driven": 27000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
        "age": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_lakhs" in data
    assert "currency" in data
    assert isinstance(data["predicted_price_lakhs"], float)
