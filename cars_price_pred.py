import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Ignore warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_data(file_path):
    """Loads data and creates initial features like car age."""
    dt = pd.read_csv(file_path)
    
    # Feature engineering: car age
    date_year = datetime.datetime.now()
    dt['age'] = date_year.year - dt['Year']
    dt.drop(['Year', 'Car_Name'], axis=1, inplace=True)
    
    return dt

def handle_outliers(df):
    """Handles outliers based on specific project thresholds and IQR."""
    # Removing extreme outliers manually
    df = df[~(df['Selling_Price'] >= 33.0) & (df['Selling_Price'] <= 35.0)]
    
    # IQR Based Outlier Removal
    q1 = df['Selling_Price'].quantile(0.25)
    q3 = df['Selling_Price'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    df = df[df['Selling_Price'] < upper_bound]
    
    return df

def regression_report(y_true, y_pred, name):
    """Prints a standard regression performance report."""
    print(f"{name} Regression Report:")
    print(f" R2 Score: {r2_score(y_true, y_pred):.4f}")
    print(f" MAE:      {mean_absolute_error(y_true, y_pred):.4f}")
    print(f" MSE:      {mean_squared_error(y_true, y_pred):.4f}")
    print(f" RMSE:     {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}\n")

def get_preprocessor():
    """Defines the preprocessing steps using ColumnTransformer."""
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
    numerical_features = ['Present_Price', 'Kms_Driven', 'Owner', 'age']
    
    # Specified categories to ensure encoding consistency (e.g. Petrol -> 0)
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

def main():
    # 1. Load and Clean
    df = load_and_preprocess_data("car data.csv")
    
    # 2. Handle Outliers
    df = handle_outliers(df)
    
    # 3. Split Features and Target
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize Preprocessor and Model Pipelines
    preprocessor = get_preprocessor()
    
    models = {
        "Random Forest": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor()
    }
    
    predictions = {}
    
    print("Training and evaluating models with Pipelines...\n")
    for name, model in models.items():
        # Build pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict and Report
        pred = pipeline.predict(X_test)
        predictions[name] = pred
        regression_report(y_test, pred, name)
        
    # 5. Result Comparison
    comparison_df = pd.DataFrame({'Actual': y_test})
    for name, pred in predictions.items():
        comparison_df[f'Predicted_{name.replace(" ", "_")}'] = pred
        
    print("Comparison of actual vs predicted (first 5 rows):")
    print(comparison_df.head())
    
    # 6. Visualization
    scores = {name: r2_score(y_test, pred) for name, pred in predictions.items()}
    
    plt.figure(figsize=(12, 7))
    plt.bar(scores.keys(), scores.values(), color=['royalblue', 'orange', 'forestgreen', 'crimson'])
    plt.title('Performance Comparison of Model Pipelines (R2 Score)')
    plt.ylabel('R2 Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
