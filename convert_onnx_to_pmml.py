from nyoka import lgb_to_pmml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Load the trained LightGBM model
with open("lightgbm_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

# Extract feature names correctly
feature_names = lgb_model.feature_name()  # Call as a function

# Load training data for fitting scaler (replace with actual training data)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Ensure scaler is fitted
if not hasattr(scaler, "mean_"):
    raise ValueError("StandardScaler must be fitted before exporting to PMML!")

# Wrap the model in a Pipeline
pipeline = Pipeline([
    ("scaler", scaler),  # Use the already fitted scaler
    ("model", lgb_model)
])

# Convert the pipeline to PMML
lgb_to_pmml(pipeline, feature_names, target_name="isFraud", pmml_f_name="lightgbm_model.pmml")

print("PMML model saved as lightgbm_model.pmml")
