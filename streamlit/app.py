import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from google.cloud import storage
from sklearn.preprocessing import StandardScaler

# Function to fetch data from GCS bucket
def fetch_data_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    with blob.open("rb") as f:
        data = pickle.load(f)
    return data

# Load the trained XGBoost model
def load_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model

# Main Streamlit App
def main():
    st.title("3-Day Weather Forecast")
    st.write("This app predicts the next 3 days' apparent maximum temperature using a trained XGBoost model.")

    # Configuration for GCS paths
    bucket_name = "clima-smart-data-collection"
    model_blob_name = "assets/daily_models/daily_best_models.json"
    scaler_blob_name = "assets/daily_scaler/daily_scaler.pkl"
    test_data_blob_name = "assets/daily_data_splits/X_test.pkl"
    test_target_blob_name = "assets/daily_data_splits/y_test.pkl"

    try:
        # Load the XGBoost model from GCS
        model_path = "/tmp/daily_best_model.json"
        storage.Client().bucket(bucket_name).blob(model_blob_name).download_to_filename(model_path)
        model = load_model(model_path)
        st.success("Model loaded successfully!")

        # Load the scaler from GCS
        scaler_blob_path = "/tmp/daily_scaler.pkl"
        storage.Client().bucket(bucket_name).blob(scaler_blob_name).download_to_filename(scaler_blob_path)
        with open(scaler_blob_path, "rb") as f:
            scaler_features = pickle.load(f)

        st.success("Scaler loaded successfully!")

        # Fetch test data and target from GCS
        X_test = fetch_data_from_gcs(bucket_name, test_data_blob_name)
        y_test = fetch_data_from_gcs(bucket_name, test_target_blob_name)

        st.write("Test data fetched successfully!")

        # Prepare test data for prediction
        X_test_scaled = scaler_features.transform(X_test)
        dtest = xgb.DMatrix(X_test_scaled)

        # Make predictions
        y_pred = model.predict(dtest)

        # Inverse transform predictions (if required)
        st.write("Predictions made successfully!")
        st.header("Prediction Results")
        
        # Display predictions for the next 3 days
        forecast = pd.DataFrame({
            "Day": [f"Day {i+1}" for i in range(len(y_pred))],
            "Predicted Apparent Temperature Max": y_pred
        })
        st.table(forecast)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

if _name_ == "_main_":
    main()