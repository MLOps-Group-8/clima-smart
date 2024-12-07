import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from google.cloud import storage
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure GCS
BUCKET_NAME = "clima-smart-data-collection"
DAILY_DATA_PATH = 'weather_data/daily_weather_data.csv'
HOURLY_DATA_PATH = 'weather_data/hourly_weather_data.csv'
MODEL_DIR_DAILY = "models/daily/"
MODEL_DIR_HOURLY = "models/hourly/"
DATE_FEATURES_DAILY = ["month", "day_of_year", "week_of_year", "is_weekend"]
DATE_FEATURES_HOURLY = ["hour", "month", "day_of_year", "week_of_year", "is_weekend"]

# Target features with min and max values
TARGET_FEATURES_DAILY = {
    'apparent_temperature_max': {'min': -10, 'max': 40},
}
TARGET_FEATURES_HOURLY = {
    'apparent_temperature': {'min': -10, 'max': 40},
    'precipitation': {'min': 0, 'max': 10},
    'rain': {'min': 0, 'max': 20},
}

# Fetch min and max values for normalization
@st.cache_resource
def get_min_max(file_path):
    """Fetch min and max values for specific target columns from a CSV file stored in GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    local_path = f"/tmp/{file_path.split('/')[-1]}"
    blob.download_to_filename(local_path)

    data = pd.read_csv(local_path)
    min_max = {}
    for column in TARGET_FEATURES_DAILY.keys() | TARGET_FEATURES_HOURLY.keys():
        if column in data.columns:
            min_max[column] = {'min': data[column].min(), 'max': data[column].max()}
    return min_max

# Fetch min/max for daily and hourly data
MIN_MAX_DAILY = get_min_max(DAILY_DATA_PATH)
MIN_MAX_HOURLY = get_min_max(HOURLY_DATA_PATH)

# Reverse normalization function
def reverse_normalization(value, min_val, max_val):
    """Reverse normalization to get the original scale."""
    return (value * (max_val - min_val)) + min_val

# Load Models from GCS
@st.cache_resource
def load_models(model_dir, _target_features):
    """Load XGBoost models from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    models = {}

    for target in _target_features:
        model_path = f"{model_dir}{target}_model.json"
        local_model_path = f"/tmp/{target}_model.json"
        blob = bucket.blob(model_path)
        blob.download_to_filename(local_model_path)

        model = xgb.XGBRegressor()
        model.load_model(local_model_path)
        models[target] = model

    return models

# Generate Daily Predictions
def predict_daily_weather(models):
    """Generate 7-day daily weather predictions."""
    today = datetime.now().date()
    future_dates = [today + timedelta(days=i) for i in range(7)]
    data = pd.DataFrame({
        "date": future_dates,
        "month": [date.month for date in future_dates],
        "day_of_year": [date.timetuple().tm_yday for date in future_dates],
        "week_of_year": [date.isocalendar().week for date in future_dates],
        "is_weekend": [1 if date.weekday() >= 5 else 0 for date in future_dates],
    })

    predictions = {}
    for target, limits in MIN_MAX_DAILY.items():
        normalized_preds = models[target].predict(data[DATE_FEATURES_DAILY])
        predictions[target] = [
            reverse_normalization(value, limits['min'], limits['max']) for value in normalized_preds
        ]

    for target in MIN_MAX_DAILY:
        data[target] = predictions[target]

    # Convert apparent_temperature_max to Fahrenheit
    if 'apparent_temperature_max' in data:
        data['apparent_temperature_max'] = data['apparent_temperature_max'] * 1.8 + 32

    return data

# Generate Hourly Predictions
def predict_hourly_weather(models, date):
    """Generate hourly weather predictions for a specific date."""
    date = pd.to_datetime(date)
    data = pd.DataFrame({
        "hour": list(range(24)),
        "month": [date.month] * 24,
        "day_of_year": [date.timetuple().tm_yday] * 24,
        "week_of_year": [date.isocalendar().week] * 24,
        "is_weekend": [1 if date.weekday() >= 5 else 0] * 24,
    })

    predictions = {}
    for target, limits in MIN_MAX_HOURLY.items():
        normalized_preds = models[target].predict(data[DATE_FEATURES_HOURLY])
        predictions[target] = [
            reverse_normalization(value, limits['min'], limits['max']) for value in normalized_preds
        ]

    for target in MIN_MAX_HOURLY:
        data[target] = predictions[target]

    return data

# Streamlit UI
def main():
    # App title
    st.title("Interactive Weather Forecasting Application")
    st.write("Select a day for daily predictions or click on a date to view hourly forecasts.")

    # Load models
    st.sidebar.header("Model Information")
    st.sidebar.write("Loading models from Google Cloud Storage...")
    daily_models = load_models(MODEL_DIR_DAILY, list(MIN_MAX_DAILY.keys()))
    hourly_models = load_models(MODEL_DIR_HOURLY, list(MIN_MAX_HOURLY.keys()))
    st.sidebar.success("Models Loaded Successfully!")

    # Date selector for daily predictions
    st.write("### Select a Date for Weather Forecast")
    today = datetime.now().date()
    selected_date = st.date_input("Select a date:", min_value=today, max_value=today + timedelta(days=6))

    # Daily Forecast
    st.write("### Daily Weather Forecast")
    daily_predictions = predict_daily_weather(daily_models)

    # Highlight selected date
    daily_predictions["Selected"] = daily_predictions["date"] == pd.Timestamp(selected_date)

    # Display predictions in a table
    st.write("#### Daily Predictions")
    st.dataframe(daily_predictions)

    # Plot daily predictions
    st.write("#### Interactive 7-Day Weather Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    for target in MIN_MAX_DAILY:
        ax.plot(daily_predictions["date"], daily_predictions[target], label=target, marker="o")
    ax.set_title("7-Day Weather Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Values")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Hourly Forecast for Selected Date
    if st.button(f"View Hourly Predictions for {selected_date}"):
        st.write(f"### Hourly Predictions for {selected_date}")
        hourly_predictions = predict_hourly_weather(hourly_models, selected_date)

        # Display hourly predictions
        st.write("#### Hourly Predictions")
        st.dataframe(hourly_predictions)

        # Plot hourly predictions
        st.write("#### Interactive Hourly Weather Chart")
        fig, ax = plt.subplots(figsize=(12, 6))
        for target in MIN_MAX_HOURLY:
            ax.plot(hourly_predictions["hour"], hourly_predictions[target], label=target, marker="o")
        ax.set_title(f"Hourly Weather Forecast for {selected_date}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Values")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
