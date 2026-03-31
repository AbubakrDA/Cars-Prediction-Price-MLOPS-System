import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import os

# Deterministic setup for feature engineering (Year basis)
REFERENCE_YEAR = 2024

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes deterministic features from input DataFrame."""
    df = df.copy()
    if 'Year' in df.columns:
        df['age'] = REFERENCE_YEAR - df['Year']
        df = df.drop(['Year'], axis=1)
    if 'Car_Name' in df.columns:
        df = df.drop(['Car_Name'], axis=1)
    return df

def load_data(file_path: str) -> pd.DataFrame:
    """Loads CSV data and applies feature engineering."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset {file_path} not found.")
    df = pd.read_csv(file_path)
    return compute_features(df)

def get_preprocessor() -> ColumnTransformer:
    """Defines deterministic preprocessing (StandardScaler, OrdinalEncoder)."""
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
    numerical_features = ['Present_Price', 'Kms_Driven', 'Owner', 'age']
    
    # We strictly enforce known categories, and if unknown categories appear inference will fail explicitly (honest fail)
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
