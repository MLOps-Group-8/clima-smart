import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Function to fetch the data from CSV file (could be from GCS or local)
def fetch_data_from_csv(file_path):
    df = pd.read_csv(daily)
    return df

# Load the trained XGBoost model
def load_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model

# Main Streamlit App
def main():
    st.title("3-Day Weather Forecast")
    st.write("This app predicts the next 3 days' apparent maximum temperature using a trained XGBoost model.")

    # Model path
    model_path = "daily_best_model.pkl"

    try:
        # Load the XGBoost model
        model = load_model(model_path)
        st.success("Model loaded successfully!")

        # Fetch real-time test data from CSV file
        # Replace with your GCS or local CSV path
        csv_file_path = "engineered_hourly_data.csv"  # Update this path
        test_data = fetch_data_from_csv(csv_file_path)

        # Display the real-time fetched data
        st.write("Fetched Real-Time Weather Data:")
        st.dataframe(test_data)

        # Initialize scaler for features (same as training setup)
        scaler_features = StandardScaler()
        scaler_features.fit_transform(np.array([[20, 10, 15, 2, 1, 10]]))  # Dummy fit for reproducibility

        # Prepare data for prediction
        X_test = test_data[['temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 
                             'rain_sum', 'showers_sum', 'daylight_duration']]  # Select relevant columns
        X_test_scaled = scaler_features.transform(X_test)
        dtest = xgb.DMatrix(X_test_scaled)

        # Make predictions
        y_pred = model.predict(dtest)

        # Display predictions
        st.header("Next 3 Days Forecast")
        forecast = pd.DataFrame({
            "Day": ["Day 1", "Day 2", "Day 3"],
            "Predicted Apparent Temperature Max": y_pred
        })
        st.table(forecast)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
