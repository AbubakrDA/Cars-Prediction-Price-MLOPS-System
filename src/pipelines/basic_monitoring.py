import pandas as pd
import os
import datetime
import json
from scipy.stats import ks_2samp

# Simple Statistical Monitoring System replacing Heavy Evidently Dep
# Computes the Kolmogorov-Smirnov test for numerical features to test if 
# train and inference data distributions diverge significantly over time.

def generate_drift_report():
    reference_data_path = "car data.csv"
    current_data_path = "inference_logs.csv"

    print("--- Basic Statistical Drift Monitoring System ---")
    if not os.path.exists(current_data_path):
        print(f"Skipping: No inference logs found at {current_data_path}. API has not received traffic yet.")
        return

    print("Loading Reference Data (Training Distribution)...")
    try:
        ref_df = pd.read_csv(reference_data_path)
    except Exception as e:
        print(f"Error loading reference data: {e}")
        return

    print("Loading Current Data (Inference Distribution)...")
    try:
        cur_df = pd.read_csv(current_data_path)
    except Exception as e:
        print(f"Error loading inference data: {e}")
        return

    # Deterministic matching
    if 'Year' in ref_df.columns:
        ref_df['age'] = 2024 - ref_df['Year']

    numerical_features = ['Present_Price', 'Kms_Driven', 'age']
    
    print(f"Comparing Reference ({len(ref_df)} rows) with Current ({len(cur_df)} rows)...")
    
    drift_results = {}
    drift_detected = False
    p_value_threshold = 0.05 # 95% confidence bounds

    for feature in numerical_features:
        if feature in ref_df.columns and feature in cur_df.columns:
            # Dropna just in case
            ref_data = ref_df[feature].dropna()
            cur_data = cur_df[feature].dropna()
            
            # Real KS-Test statistical comparison
            stat, p_value = ks_2samp(ref_data, cur_data)
            
            is_drifting = bool(p_value < p_value_threshold)
            drift_results[feature] = {
                "ks_statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "drift_detected": is_drifting
            }
            if is_drifting:
                drift_detected = True

    final_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "reference_rows": len(ref_df),
        "current_rows": len(cur_df),
        "overall_drift_detected": drift_detected,
        "feature_metrics": drift_results
    }

    print("-> Statistical Drift Report Generated:")
    print(json.dumps(final_report, indent=2))
    print("-> Saved to: drift_report.json")

    with open('drift_report.json', 'w') as f:
        json.dump(final_report, f, indent=4)

if __name__ == "__main__":
    generate_drift_report()
