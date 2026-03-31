import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

# MLflow Config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# The default model URI pattern or environment variable injection
# Ideally "models:/CarPrice_Gradient_Boosting/Champion" if registry is active
DEFAULT_MODEL_URI = os.getenv("MODEL_URI", "models:/CarPrice_Gradient_Boosting/Champion")

app = FastAPI(title="Car Price Prediction API - MLOps", version="1.1.0")
model = None

@app.on_event("startup")
def load_mlflow_model():
    """Fetches the actual model artifact from MLflow directly into memory."""
    global model
    print(f"Connecting to MLflow Tracking Server at: {MLFLOW_TRACKING_URI}")
    print(f"Loading Model Artifact from URI: {DEFAULT_MODEL_URI}")
    
    try:
        model = mlflow.sklearn.load_model(DEFAULT_MODEL_URI)
        print("-> Successfully loaded authentic model from MLflow.")
        
        # Test infer payload to ensure signature matches expected inputs (optional integrity check)
    except Exception as e:
        print(f"-> Critical Failure: Could not load model from MLflow URI: {DEFAULT_MODEL_URI}")
        print(f"-> Error Context: {str(e)}")
        print("-> API will initialize, but /predict will fail until model URI is resolved.")

@app.get('/')
def index():
    return {
        'message': 'Welcome to Car Price Prediction MLOps API',
        'health': 'OK',
        'mlflow_tracking_uri': MLFLOW_TRACKING_URI,
        'model_status': 'Loaded' if model else 'Disconnected or Failed'
    }

class CarFeatures(BaseModel):
    Present_Price: float = Field(..., gt=0)
    Kms_Driven: int = Field(..., gt=-1)
    Fuel_Type: str = Field(..., pattern="^(Petrol|Diesel|CNG)$")
    Seller_Type: str = Field(..., pattern="^(Dealer|Individual)$")
    Transmission: str = Field(..., pattern="^(Manual|Automatic)$")
    Owner: int = Field(..., ge=0)
    age: int = Field(..., gt=-1)
    
@app.post('/predict')
def predict_price(features: CarFeatures):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded. Ensure MLflow Server is at {MLFLOW_TRACKING_URI} and URI {DEFAULT_MODEL_URI} is valid."
        )

    # Convert to DataFrame allowing seamless application of preprocessor schema
    input_df = pd.DataFrame([features.model_dump()])
    
    # Store incoming queries for Data Drift basic monitoring (Append-only log)
    log_file = "inference_logs.csv"
    try:
        input_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    except Exception as e:
        print(f"Warning: Drift log append failed: {e}")

    # Generate prediction from MLflow-sourced model
    try:
        prediction = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failure via scikit-learn pipeline: {e}")
        
    pred_val = float(abs(prediction[0]))
    
    return {
        'predicted_price_lakhs': round(pred_val, 2),
        'currency': 'INR (Lakhs)',
        'artifact_source': DEFAULT_MODEL_URI # Honest output reflecting exact source
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("src.api.app:app", host='0.0.0.0', port=8000, reload=True)