# Car Price Prediction 🚗💰

**Car Price Prediction** is a simple machine learning project that predicts used car prices using regression models. The repository includes an exploratory notebook, a trained model evaluation report, a small dataset, and a FastAPI app for serving predictions.

---

## 🔍 Project Overview

- **Goal:** Predict the selling price of used cars based on features such as year, mileage, make/model, engine size, and more.
- **Files of interest:**
  - `car data.csv` — dataset used for training and evaluation
  - `cars_price_pred.ipynb` — Jupyter notebook with data exploration, preprocessing, model training, and evaluation
  - `regression_report.csv` — evaluation metrics and error analysis
  - `DeployfastApi.py` — FastAPI app to serve predictions
  - `dockerfile` — Dockerfile to containerize the API
  - `requirements.txt` — Python dependencies

---

## ⚙️ Setup & Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd "Car Price Prediction"
   ```

2. (Recommended) Create and activate a virtual environment:

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Run the analysis notebook

Open `cars_price_pred.ipynb` in Jupyter or VS Code and run the cells to reproduce data exploration, preprocessing, training, and evaluation steps.

### Run the API locally

1. Start the API:

```bash
python DeployfastApi.py
```

2. By default, the app will be served (e.g., at `http://127.0.0.1:8000`). Use the interactive docs at `/docs` to test prediction endpoints.

### Docker (optional)

Build and run the Docker container:

```bash
docker build -t car-price-api -f dockerfile .
docker run -p 8000:8000 car-price-api
```

---

## 🧪 Model & Evaluation

- The notebook contains the training pipeline and model selection.
- Results and metrics (MAE, RMSE, R^2) are available in `regression_report.csv`.
- Tips to improve performance: feature engineering, hyperparameter tuning, and using ensemble models.

---

## 🛠️ Contributing

Contributions are welcome — please open an issue or submit a pull request with improvements, bug fixes, or new features.

---

## 📄 License

This project is provided under the MIT License (or replace with your chosen license).

---

## ✉️ Contact

For questions or feedback, open an issue or reach out via the repository contact information.


**Happy modeling!** ✅
