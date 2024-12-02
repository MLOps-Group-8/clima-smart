import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the trained XGBoost model
def load_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model

# Load sample test data for prediction
def load_test_data(scaler_features):
    # Generate mock data resembling the test data structure
    test_data = {
        'temperature_2m_max': [20, 21, 19],
        'temperature_2m_min': [10, 9, 8],
        'apparent_temperature_min': [15, 14, 13],
        'rain_sum': [2, 0, 3],
        'showers_sum': [1, 0, 2],
        'daylight_duration': [10, 11, 9],
    }
    df = pd.DataFrame(test_data)
    df_scaled = scaler_features.transform(df)
    return df_scaled

# Main Streamlit App
def main():
    st.title("3-Day Weather Forecast")
    st.write("This app predicts the next 3 days' apparent maximum temperature using a trained XGBoost model.")

    # Model path
    model_path = "xgb_model_apparent_temperature_max.json"

    try:
        # Load model
        model = load_model(model_path)
        st.success("Model loaded successfully!")

        # Initialize scaler for features (same as training setup)
        scaler_features = StandardScaler()
        scaler_features.fit_transform(np.array([[20, 10, 15, 2, 1, 10]]))  # Dummy fit for reproducibility

        # Load test data
        X_test_sample = load_test_data(scaler_features)
        dtest_sample = xgb.DMatrix(X_test_sample)

        # Make predictions
        y_pred_sample = model.predict(dtest_sample)

        # Display predictions
        st.header("Next 3 Days Forecast")
        forecast = pd.DataFrame({
            "Day": ["Day 1", "Day 2", "Day 3"],
            "Predicted Apparent Temperature Max": y_pred_sample
        })
        st.table(forecast)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
