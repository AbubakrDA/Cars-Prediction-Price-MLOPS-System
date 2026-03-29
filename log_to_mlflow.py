import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os
import sys

def log_existing_model():
    print("Initializing MLflow logging process...")
    try:
        # Set the experiment name
        mlflow.set_experiment("Car_Price_Prediction")
        print("Experiment 'Car_Price_Prediction' set.")

        model_path = "car_price_prediction_model.pkl"
        report_path = "regression_report.csv"

        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found.")
            return

        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        
        best_model_metrics = None
        if os.path.exists(report_path):
            print(f"Reading metrics from {report_path}...")
            metrics_df = pd.read_csv(report_path)
            if 'Model' in metrics_df.columns:
                matches = metrics_df[metrics_df['Model'] == 'Gradient Boosting']
                if not matches.empty:
                    best_model_metrics = matches.iloc[0]
                    print("Found metrics for Gradient Boosting.")
        else:
            print(f"Report file {report_path} not found. Skipping metrics logging.")

        print("Starting MLflow run...")
        with mlflow.start_run(run_name="Migration_Run") as run:
            print(f"Run started with ID: {run.info.run_id}")
            
            # Log model
            print("Logging scikit-learn model to MLflow...")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="CarPriceModel"
            )
            print("Model logged to MLflow artifacts.")

            # Log metrics if available
            if best_model_metrics is not None:
                print("Logging metrics...")
                mlflow.log_metric("R2_Score", float(best_model_metrics['R2 Score']))
                mlflow.log_metric("MAE", float(best_model_metrics['MAE']))
                mlflow.log_metric("MSE", float(best_model_metrics['MSE']))
                mlflow.log_metric("RMSE", float(best_model_metrics['RMSE']))

            print(f"SUCCESS: Model logged from {model_path} to MLflow.")

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    log_existing_model()
