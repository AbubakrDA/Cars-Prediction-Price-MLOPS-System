import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import datetime
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score, median_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# 1. Set MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_and_preprocess_data(file_path):
    """Loads data and creates initial features."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset {file_path} not found.")
    
    df = pd.read_csv(file_path)
    date_year = datetime.datetime.now()
    df['age'] = date_year.year - df['Year']
    df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
    return df

def get_preprocessor():
    """Defines the Level 1 preprocessing (StandardScaler, OrdinalEncoder)."""
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
    numerical_features = ['Present_Price', 'Kms_Driven', 'Owner', 'age']
    
    fuel_categories = ['Petrol', 'Diesel', 'CNG']
    seller_categories = ['Dealer', 'Individual']
    transmission_categories = ['Manual', 'Automatic']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OrdinalEncoder(categories=[fuel_categories, seller_categories, transmission_categories]), categorical_features)
        ]
    )
    return preprocessor

def train_and_log_models():
    # Set Experiment
    experiment_name = "Car_Price_Prediction_Pipelines"
    mlflow.set_experiment(experiment_name)
    
    # Load Data
    try:
        df = load_and_preprocess_data("car data.csv")
    except FileNotFoundError as e:
        print(e)
        return

    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = get_preprocessor()
    
    models = {
        "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient_Boosting": GradientBoostingRegressor(random_state=42),
        "Linear_Regression": LinearRegression(),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }
    
    print(f"Connected to MLflow at: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {experiment_name}\n")

    best_r2 = -float("inf")
    best_pipeline = None
    best_model_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Level2_{name}"):
            # Build 2-level Pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            evs = explained_variance_score(y_test, y_pred)
            medae = median_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            # mean_squared_log_error requires non-negative values
            try:
                msle = mean_squared_log_error(y_test, np.clip(y_pred, 0, None))
            except Exception as e:
                msle = None
                print(f"  Could not calculate MSLE for {name}: {e}")
            
            # Log Parameters
            mlflow.log_param("model_name", name)
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", model.n_estimators)
            
            # Log Metrics
            mlflow.log_metric("R2_Score", r2)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("Explained_Variance", evs)
            mlflow.log_metric("Median_AE", medae)
            mlflow.log_metric("MAPE", mape)
            if msle is not None:
                mlflow.log_metric("MSLE", msle)
            
            # Log the whole 2-level Pipeline
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="pipeline_model",
                registered_model_name=f"CarPrice_{name}"
            )
            
            # Track the overall best model for local artifact generation
            if r2 > best_r2:
                best_r2 = r2
                best_pipeline = pipeline
                best_model_name = name
            
            print(f"Logged {name}: R2={r2:.4f}, RMSE={rmse:.4f}")

    # After the loop, save the best model to a local file for Docker AWS deployment
    if best_pipeline:
        print(f"\nSaving best model ({best_model_name} with R2={best_r2:.4f}) to car_price_prediction_model.pkl for offline deployment...")
        joblib.dump(best_pipeline, 'car_price_prediction_model.pkl')
        print("Model saved successfully.")

if __name__ == "__main__":
    train_and_log_models()
