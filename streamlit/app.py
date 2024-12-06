import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
import tempfile
import os

# Google Cloud Storage Setup
BUCKET_NAME = "clima-smart-data-collection"

# Ensure the credentials are set up
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not credentials_path:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")


def fetch_data_from_gcs(bucket_name, file_path):
    """Fetch file from GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        return pd.read_csv(temp_file.name)

def load_model_from_gcs(bucket_name, model_path):
    """Load model from GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        model = xgb.Booster()
        model.load_model(temp_file.name)
        return model

# Main Streamlit App
def main():

    st.set_page_config(page_title="3-Day Weather Forecast", layout="wide")
    st.title("🌤️ 3-Day Weather Forecast")
    st.write("This app predicts the next 3 days' apparent maximum temperature using a trained XGBoost model.")

    # Sidebar Configuration
    st.sidebar.header("Configuration")
    model_path = st.sidebar.text_input("Model Path in GCS", "assets/daily_models/daily_best_model.json")
    csv_file_path = st.sidebar.text_input("CSV Path in GCS", "weather_data/engineered_daily_data.csv")

    try:
        # Fetch and load model
        st.sidebar.info("Fetching and loading model...")
        model = load_model_from_gcs(BUCKET_NAME, model_path)
        st.sidebar.success("Model loaded successfully!")

        # Fetch CSV data
        st.sidebar.info("Fetching real-time data...")
        test_data = fetch_data_from_gcs(BUCKET_NAME, csv_file_path)
        st.sidebar.success("Data fetched successfully!")

        # Display the fetched data
        st.subheader("Fetched Real-Time Weather Data:")
        st.dataframe(test_data)

        # Initialize scaler (ensure consistency with training setup)
        scaler_features = StandardScaler()
        scaler_features.fit(np.array([[20, 10, 15, 2, 1, 10]]))  # Dummy fit for reproducibility

        # Prepare data for prediction
        X_test = test_data[['temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 
                             'rain_sum', 'showers_sum', 'daylight_duration']]
        X_test_scaled = scaler_features.transform(X_test)
        dtest = xgb.DMatrix(X_test_scaled)

        # Make predictions
        y_pred = model.predict(dtest)

        # Display predictions with a chart
        st.subheader("📊 Next 3 Days Forecast")
        forecast = pd.DataFrame({
            "Day": ["Day 1", "Day 2", "Day 3"],
            "Predicted Apparent Temperature Max": y_pred
        })
        st.table(forecast)

        # Visualization
        st.line_chart(forecast.set_index("Day"))

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
