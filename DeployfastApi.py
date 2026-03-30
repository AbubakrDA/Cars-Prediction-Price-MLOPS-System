# Deploying FastAPI Application with Offline Model Artifact
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Car Price Prediction API (Production)")

# 1. Load the best-performing model (Pipeline) from the local artifact
model_path = "car_price_prediction_model.pkl"
print(f"Loading offline model from {model_path}...")
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    print("Model loaded successfully from offline artifact.")
except Exception as e:
    print(f"Error loading offline model: {e}")
    model = None

@app.get('/')
def index():
    return {
        'message': 'Welcome to Car Price Prediction API',
        'status': 'Model Loaded' if model else 'Model Not Loaded'
    }

# 3. Enhanced Schema: Categorical values are now strings (Petrol, Dealer, Manual, etc.)
class CarFeatures(BaseModel):
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: str      # Changed to str
    Seller_Type: str    # Changed to str
    Transmission: str   # Changed to str
    Owner: int
    age: int
    
@app.post('/predict')
def predict_price(features: CarFeatures):
    if model is None:
        return {'error': 'Model not loaded. Ensure MLflow server is running.'}

    # 4. Prepare data as DataFrame (required for 2-level pipeline)
    input_df = pd.DataFrame([features.model_dump()])
    
    # 5. Data Drift & Decay Logging: Save the incoming query distribution
    log_file = "inference_logs.csv"
    try:
        input_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    except Exception as e:
        print(f"Warning: Failed to log inference for Data Drift tracking: {e}")

    # Generate prediction from the pipeline
    prediction = model.predict(input_df)
    
    # Optional logic: For MSLE protection, ensure prediction is positive
    pred_val = abs(prediction[0])
    
    return {
        'predicted_price_lakhs': float(round(pred_val, 2)),
        'currency': 'INR (Lakhs)',
        'note': 'Model retrieved from MLflow Tracking'
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)