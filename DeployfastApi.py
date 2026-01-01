# Deploying FastAPI Application
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()
model = pickle.load(open('car_price_prediction_model.pkl', 'rb'))
@app.get('/')
def index():
    return {'message': 'Welcome to Car Price Prediction API'}
# post method to predict car price
# Example input: [Year, Present_Price, Kms_Driven, Owner, Fuel_Type, Seller_Type, Transmission, Mileage, Engine, Power, Seats]
class CarFeatures(BaseModel):
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: int
    Seller_Type: int
    Transmission: int
    Owner: int
    age: int
    
@app.post('/predict')
def predict_price(features: CarFeatures):
    input_data = np.array([[features.Present_Price, 
                            features.Kms_Driven, 
                            features.Fuel_Type,
                            features.Seller_Type, 
                            features.Transmission,
                            features.Owner,
                            features.age]]) 
    
    
    prediction = model.predict(input_data)
    return {'predicted_price': float(prediction[0])}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
    