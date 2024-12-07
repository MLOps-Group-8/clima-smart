import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from google.cloud import storage
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure GCS
BUCKET_NAME = "clima-smart-data-collection"
MODEL_DIR_DAILY = "models/daily/"
MODEL_DIR_HOURLY = "models/hourly/"
DATE_FEATURES_DAILY = ["month", "day_of_year", "week_of_year", "is_weekend"]
DATE_FEATURES_HOURLY = ["hour", "month", "day_of_year", "week_of_year", "is_weekend"]

# Load Models from GCS
@st.cache_resource
def load_models(model_dir, target_features):
    """Load XGBoost models from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    models = {}

    for target in target_features:
        model_path = f"{model_dir}{target}_model.json"
        local_model_path = f"/tmp/{target}_model.json"
        blob = bucket.blob(model_path)
        blob.download_to_filename(local_model_path)

        model = xgb.XGBRegressor()
        model.load_model(local_model_path)
        models[target] = model

    return models

# Generate Daily Predictions
def predict_daily_weather(models, target_features):
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
    for target in target_features:
        predictions[target] = models[target].predict(data[DATE_FEATURES_DAILY])

    data["Apparent Temperature (°C)"] = predictions["apparent_temperature_max"]
    data["Humidity (%)"] = predictions["humidity"]
    data["Wind Speed (m/s)"] = predictions["wind_speed"]

    return data

# Generate Hourly Predictions
def predict_hourly_weather(models, date, target_features):
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
    for target in target_features:
        predictions[target] = models[target].predict(data[DATE_FEATURES_HOURLY])

    data["Apparent Temperature (°C)"] = predictions["apparent_temperature_max"]
    data["Humidity (%)"] = predictions["humidity"]
    data["Wind Speed (m/s)"] = predictions["wind_speed"]

    return data

# Streamlit UI
def main():
    # App title
    st.title("Interactive Weather Forecasting Application")
    st.write("Select a day for daily predictions or click on a date to view hourly forecasts.")

    # Load models
    target_features = ["apparent_temperature_max", "humidity", "wind_speed"]
    st.sidebar.header("Model Information")
    st.sidebar.write("Loading models from Google Cloud Storage...")
    daily_models = load_models(MODEL_DIR_DAILY, target_features)
    hourly_models = load_models(MODEL_DIR_HOURLY, target_features)
    st.sidebar.success("Models Loaded Successfully!")

    # Date selector for daily predictions
    st.write("### Select a Date for Weather Forecast")
    today = datetime.now().date()
    selected_date = st.date_input("Select a date:", min_value=today, max_value=today + timedelta(days=6))

    # Daily Forecast
    if selected_date:
        st.write(f"### Daily Weather Forecast for the Next 7 Days")
        daily_predictions = predict_daily_weather(daily_models, target_features)

        # Highlight selected date
        daily_predictions["Selected"] = daily_predictions["date"] == pd.Timestamp(selected_date)

        # Display predictions in a table
        st.write("#### Daily Predictions")
        st.dataframe(daily_predictions)

        # Plot daily predictions
        st.write("#### Interactive 7-Day Weather Chart")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_predictions["date"], daily_predictions["Apparent Temperature (°C)"], label="Temperature (°C)", marker="o")
        ax.plot(daily_predictions["date"], daily_predictions["Humidity (%)"], label="Humidity (%)", marker="s")
        ax.plot(daily_predictions["date"], daily_predictions["Wind Speed (m/s)"], label="Wind Speed (m/s)", marker="^")
        ax.set_title("7-Day Weather Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Values")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Hourly Forecast for Selected Date
        if daily_predictions["Selected"].any():
            st.write(f"### Hourly Predictions for {selected_date}")
            hourly_predictions = predict_hourly_weather(hourly_models, selected_date, target_features)

            # Display hourly predictions
            st.write("#### Hourly Predictions")
            st.dataframe(hourly_predictions)

            # Plot hourly predictions
            st.write("#### Interactive Hourly Weather Chart")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(hourly_predictions["hour"], hourly_predictions["Apparent Temperature (°C)"], label="Temperature (°C)", marker="o")
            ax.plot(hourly_predictions["hour"], hourly_predictions["Humidity (%)"], label="Humidity (%)", marker="s")
            ax.plot(hourly_predictions["hour"], hourly_predictions["Wind Speed (m/s)"], label="Wind Speed (m/s)", marker="^")
            ax.set_title(f"Hourly Weather Forecast for {selected_date}")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Values")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
