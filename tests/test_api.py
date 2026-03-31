from fastapi.testclient import TestClient
from src.api.app import app
import pytest

client = TestClient(app)

def test_index_health():
    """Verify that the API root is accessible and returns health info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["health"] == "OK"
    assert "mlflow_tracking_uri" in data

def test_invalid_schema():
    """Verify that pydantic strict schema parsing rejects invalid floats/strings."""
    payload = {
        "Present_Price": -5.0, # Cannot be negative
        "Kms_Driven": 27000,
        "Fuel_Type": "InvalidFuel", # Must fail enum regex
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
        "age": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
    
    errors = response.json()["detail"]
    # We expect multiple errors (negative price and bad fuel string)
    error_fields = [e["loc"][-1] for e in errors]
    assert "Present_Price" in error_fields
    assert "Fuel_Type" in error_fields

def test_prediction_payload_schema_success():
    """Verify inference request format (skips model testing if MLflow is offline)"""
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
    # If the model is not loaded (503) or prediction succeeds (200),
    # Ensure it's not a payload parsing failure (422)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price_lakhs" in data
        assert isinstance(data["predicted_price_lakhs"], float)
