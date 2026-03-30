# pipeline_demo.py
"""Demonstration of a two‑level ML pipeline for car price prediction.

Level 1 – Preprocessing:
    * Numerical features are scaled using ``StandardScaler``.
    * Categorical features are encoded with ``OrdinalEncoder``.

Level 2 – Modeling:
    * Several regressors (RandomForest, GradientBoosting, LinearRegression, XGBRegressor)
      are attached to the same preprocessing pipeline.
    * Each model is trained and evaluated independently, showcasing how the
      preprocessing step can be reused.

This script is intentionally self‑contained and can be run directly from the
project root. It mirrors the logic in ``cars_price_pred.py`` but isolates the
pipeline construction for clarity.
"""

import warnings
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load the CSV dataset and create the ``age`` feature.

    The original ``Year`` column is dropped because ``age`` already captures the
    information needed for the model.
    """
    df = pd.read_csv(file_path)
    # Feature engineering: car age
    current_year = datetime.datetime.now().year
    df["age"] = current_year - df["Year"]
    df.drop(["Year", "Car_Name"], axis=1, inplace=True)
    return df


def get_preprocessor() -> ColumnTransformer:
    """Return a ``ColumnTransformer`` that applies ``StandardScaler`` to numeric
    columns and ``OrdinalEncoder`` to categorical columns.
    """
    categorical_features = ["Fuel_Type", "Seller_Type", "Transmission"]
    numerical_features = ["Present_Price", "Kms_Driven", "Owner", "age"]

    # Explicit category ordering to guarantee reproducible encoding
    fuel_categories = ["Petrol", "Diesel", "CNG"]
    seller_categories = ["Dealer", "Individual"]
    transmission_categories = ["Manual", "Automatic"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OrdinalEncoder(
                    categories=[fuel_categories, seller_categories, transmission_categories]
                ),
                categorical_features,
            ),
        ]
    )
    return preprocessor


def build_model_pipeline(model) -> Pipeline:
    """Wrap a regressor with the shared preprocessing pipeline.

    Parameters
    ----------
    model: estimator instance
        Any scikit‑learn compatible regressor (e.g., ``RandomForestRegressor``).

    Returns
    -------
    Pipeline
        ``Pipeline`` with two steps: ``preprocessor`` and ``regressor``.
    """
    return Pipeline(steps=[("preprocessor", get_preprocessor()), ("regressor", model)])


def regression_report(y_true, y_pred, name: str) -> None:
    """Print a concise regression performance summary for a given model."""
    print(f"{name} Regression Report:")
    print(f"  R2 Score : {r2_score(y_true, y_pred):.4f}")
    print(f"  MAE      : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  MSE      : {mean_squared_error(y_true, y_pred):.4f}")
    print(f"  RMSE     : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}\n")


def main():
    # ---------------------------------------------------------------------
    # 1️⃣ Load data & split
    # ---------------------------------------------------------------------
    df = load_and_preprocess_data("car data.csv")
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------------------
    # 2️⃣ Define regressors
    # ---------------------------------------------------------------------
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42),
    }

    predictions = {}
    print("Training and evaluating each model with the shared preprocessing pipeline...\n")
    for name, reg in models.items():
        pipeline = build_model_pipeline(reg)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        predictions[name] = pred
        regression_report(y_test, pred, name)

    # ---------------------------------------------------------------------
    # 3️⃣ Visualise R² scores for quick comparison
    # ---------------------------------------------------------------------
    scores = {name: r2_score(y_test, pred) for name, pred in predictions.items()}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette="viridis")
    plt.title("R² Score Comparison Across Regressors (Shared Pre‑processing)")
    plt.ylim(0, 1.1)
    plt.ylabel("R² Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
