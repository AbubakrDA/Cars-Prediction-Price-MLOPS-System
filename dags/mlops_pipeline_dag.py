import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Get the absolute path to the project directory to ensure the scripts can find their files
# Assuming this DAG file is at <Project_Dir>/dags/mlops_pipeline_dag.py
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'car_price_mlops_pipeline',
    default_args=default_args,
    description='A simple ML pipeline for car price prediction',
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'car-price'],
) as dag:

    # Task 1: Train Models and Log to MLflow
    train_models = BashOperator(
        task_id='train_and_log_models',
        bash_command=f'cd "{PROJECT_DIR}" && python train_mlflow.py',
    )

    # Task 2: Monitor Data/Model Drift
    monitor_drift = BashOperator(
        task_id='monitor_model_drift',
        bash_command=f'cd "{PROJECT_DIR}" && python monitor_drift.py',
    )

    # Define simple dependencies
    train_models >> monitor_drift
