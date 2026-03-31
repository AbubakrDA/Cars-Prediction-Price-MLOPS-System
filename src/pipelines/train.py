import os
import sys

# Ensure src module is in the path when run interactively or via Airflow BashOperator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, 
                             explained_variance_score, median_absolute_error, 
                             mean_absolute_percentage_error)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.pipelines.data_prep import load_data, get_preprocessor

# 1. Set MLflow Tracking URI using ENV var falling back to local
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def train_and_log_models():
    experiment_name = "Car_Price_Prediction_Pipelines"
    mlflow.set_experiment(experiment_name)
    
    data_path = "car data.csv"
    
    try:
        # 2. Reproducible feature engineering
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(e)
        return

    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = get_preprocessor()
    
    # Keeping the models similar to original
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
    best_run_id = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Level2_{name}") as run:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            evs = explained_variance_score(y_test, y_pred)
            medae = median_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            mlflow.log_param("model_name", name)
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", model.n_estimators)
            
            mlflow.log_metric("R2_Score", r2)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("Explained_Variance", evs)
            mlflow.log_metric("MAPE", mape)
            
            # Infer signature for better downstream serving integrity
            signature = infer_signature(X_test, y_pred)
            input_example = X_test.iloc[:1]
            
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="pipeline_model",
                signature=signature,
                input_example=input_example,
                registered_model_name=f"CarPrice_{name}"
            )
            
            if r2 > best_r2:
                best_r2 = r2
                best_pipeline = pipeline
                best_model_name = name
                best_run_id = run.info.run_id
                
            print(f"Logged {name}: R2={r2:.4f}, RMSE={rmse:.4f}")

    print(f"\nTraining Complete. Best Model: {best_model_name} (R2={best_r2:.4f})")
    print(f"To serve this model via MLflow URI, use: runs:/{best_run_id}/pipeline_model")
    
    # Register Champion alias explicitly
    try:
        from mlflow.client import MlflowClient
        client = MlflowClient()
        model_reg_name = f"CarPrice_{best_model_name}"
        latest_versions = client.search_model_versions(f"name='{model_reg_name}'")
        if latest_versions:
            latest_version = max([int(v.version) for v in latest_versions])
            client.set_registered_model_alias(model_reg_name, "Champion", str(latest_version))
            print(f"Registered '{model_reg_name}' version {latest_version} as 'Champion' in MLflow Registry.")
            
            # Since relying completely on remote MLflow models can be fragile for a junior project demo,
            # we write out a single transparent hint file pointing to the latest champion run URI.
            with open("latest_champion_uri.txt", "w") as f:
                f.write(f"models:/{model_reg_name}@Champion")
    except Exception as e:
        print(f"Warning: Could not assign Champion alias (Registry might not be fully backed): {e}")

if __name__ == "__main__":
    train_and_log_models()
