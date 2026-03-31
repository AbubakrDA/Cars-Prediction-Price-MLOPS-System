# 🚗 Car Price Prediction MLOps Portfolio

This repository demonstrates an end-to-end Machine Learning pipeline for predicting used car prices. It is built to showcase a realistic, junior-to-mid-level MLOps workflow focusing on reproducibility, sensible pipeline separation, and explicit deployment mechanics.

Unlike "demo" projects, this repository enforces strict boundaries between Data Engineering, Training, and API Serving, leveraging legitimate tracking software rather than isolated `.pkl` artifacts where possible.

---

## 🏗 System Architecture & Design Choices

### Core Workflow

1.  **Data Preparation**: Deterministic feature engineering (`src/pipelines/data_prep.py`) ensuring identical transformations are applied during training and inference.
2.  **Experiment Tracking**: `src/pipelines/train.py` trains multiple models and logs hyperparameters, metrics, and serialized artifacts with input signatures directly to **MLflow**.
3.  **Inference Serving**: A FastAPI application (`src/api/app.py`) dynamically pulls the "Champion" model via MLflow URI on startup instead of relying on manually copied binaries.
4.  **Continuous Monitoring**: `src/pipelines/basic_monitoring.py` implements a real Statistical KS-Test comparing incoming traffic against training baselines to detect potential data drift (concept drift).
5.  **Orchestration**: A professional DAG via **Apache Airflow** schedules the retraining and monitoring phases asynchronously.

---

## 📁 Repository Structure

```text
.
├── src/                             # Core Python applications
│   ├── api/                         # FastAPI inference microservice
│   │   └── app.py
│   └── pipelines/                   # MLOps ML workflows
│       ├── data_prep.py             # Deterministic shared feature prep
│       ├── train.py                 # MLflow tracking and model training
│       └── basic_monitoring.py      # Statistical KS-Test drift validation
├── dags/                            # Apache Airflow orchestration DAGs
│   └── mlops_pipeline_dag.py
├── tests/                           # Pytest suite targeting API robustness
│   ├── test_api.py
│   └── test_api_client.py           # Stress-test script
├── infra/
│   └── terraform/                   # AWS App Runner config templates
├── Dockerfile                       # Minimal serving container (FastAPI only)
├── docker-compose.yaml              # Local Airflow cluster configuration
├── requirements-*.txt               # Hard-isolated dependency domains
└── README.md
```

---

## 🚦 Current Implementation Status
*An honest record of what is functioning, what is simulated, and what is planned.*

**What IS completely implemented:**
*   **MLflow Integration**: The code logs full experiments natively and the API serves the model retrieved *directly* from trackable `models:/` endpoints (or local registry paths).
*   **Dependency Hygiene**: Training dependencies (`xgboost`, `scipy`) are segregated from API serving dependencies (`fastapi`, `pydantic`), enabling highly cacheable Docker builds.
*   **Reproducibility**: Dates for features like `car age` strictly lock to `REFERENCE_YEAR=2024` avoiding non-deterministic rot over time.
*   **Basic Drift Monitoring**: We use the Kolmogorov-Smirnov check to flag shifts between training (`car data.csv`) and logged inference queries (`inference_logs.csv`).
*   **API Validation**: The `/predict` endpoint uses Pydantic's strict regex and bounds features ensuring malformed data strictly fails before entering the ML layer.

**What is PARTIALLY or NOT fully hardened (Interview Transparency):**
*   **Terraform & Remote State**: The AWS Apprunner `.tf` files exist as conceptual structure, but state locks, proper Secrets Management (Variables/KMS), and CI/CD IAM boundaries are outside the scope of this portfolio piece.
*   **DVC (Data Version Control)**: `.dvc` tracking exists locally but does not strictly validate remote pushing in the action runner.
*   **Monitoring Scale**: The `basic_monitoring.py` script replaces heavier enterprise dependencies (like Evidently AI) for portfolio footprint viability.

---

## 🚀 Setup & Execution (Local)

### 1. Model Training Workflow

Initialize your environment, train the candidate models, and log the results into MLflow.

```bash
# Set up a clean environment
python -m venv venv
# On Windows: venv\Scripts\activate | On Unix: source venv/bin/activate
pip install -r requirements-train.txt

# Run the training pipeline
python src/pipelines/train.py

# Launch the Tracking Server UI (Optional)
mlflow ui
# View experiments at http://127.0.0.1:5000
```

### 2. Standalone Inference Server
The API reads `MODEL_URI` matching the newly generated MLflow artifact.

```bash
# Isolate serving dependencies
pip install -r requirements-api.txt
# Launch API 
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Then send a payload to `http://localhost:8000/predict`:
```json
{
  "Present_Price": 5.5,
  "Kms_Driven": 27000,
  "Fuel_Type": "Petrol",
  "Seller_Type": "Dealer",
  "Transmission": "Manual",
  "Owner": 0,
  "age": 10
}
```

### 3. Continuous Orchestration (Airflow)
Local automated scheduling via official docker bindings.

```bash
pip install -r requirements-orchestration.txt
docker compose up airflow-init
docker compose up -d
```
Access `http://localhost:8080` to toggle the DAG `car_price_mlops_pipeline`.

### 4. Developer Testing
```bash
pip install -r requirements-dev.txt
pytest tests/
```

---
*Developed as a realistic representation of production MLOps practices.*
