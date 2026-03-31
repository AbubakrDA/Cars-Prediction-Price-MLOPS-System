import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# Automatically resolve paths to tolerate running anywhere
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PIPELINES_DIR = os.path.join(PROJECT_DIR, 'src', 'pipelines')

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False, # In a real environment, set to proper ops email
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30)
}

with DAG(
    'car_price_mlops_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline representing training and continuous drift monitoring',
    schedule_interval='@daily',
    catchup=False,
    tags=['core-mlops', 'car-price-predictor'],
) as dag:

    # 1. Start logic
    start_pipeline = EmptyOperator(task_id='start_pipeline')

    # 2. Train Models (Includes Data Prep phase natively)
    train_models = BashOperator(
        task_id='train_and_log_models_to_mlflow',
        bash_command=f'cd "{PROJECT_DIR}" && python "{os.path.join(PIPELINES_DIR, "train.py")}"',
    )
    
    # 3. Model gate - A simulated step where an active evaluation script would run
    # For now, we represent an empty operator gating the drift checks
    validate_model_performance = EmptyOperator(task_id='validate_metrics_gate')

    # 4. Monitor Data/Model Drift
    monitor_drift = BashOperator(
        task_id='monitor_data_and_concept_drift',
        bash_command=f'cd "{PROJECT_DIR}" && python "{os.path.join(PIPELINES_DIR, "basic_monitoring.py")}"',
    )
    
    end_pipeline = EmptyOperator(task_id='end_pipeline')

    # Explicit orchestration dependencies
    start_pipeline >> train_models >> validate_model_performance >> monitor_drift >> end_pipeline
