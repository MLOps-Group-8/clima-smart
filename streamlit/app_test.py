import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb

# Import functions to test
from app import (
    get_min_max,
    load_models,
    predict_daily_weather,
    predict_hourly_weather,
    reverse_normalization,
)


class TestWeatherForecastApp(unittest.TestCase):

    # Test: Fetch min and max values from GCS
    @patch("google.cloud.storage.Client")
    def test_get_min_max(self, mock_client):
        mock_data = pd.DataFrame({'apparent_temperature_max': [10, 20, 30]})
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Mock file download
        mock_blob.download_to_filename.side_effect = lambda filename: mock_data.to_csv(filename, index=False)

        result = get_min_max("mock_path")
        self.assertEqual(result['apparent_temperature_max']['min'], 10)
        self.assertEqual(result['apparent_temperature_max']['max'], 30)

    # Test: Load XGBoost models from GCS
    @patch("google.cloud.storage.Client")
    def test_load_models(self, mock_client):
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Mock model download
        mock_blob.download_to_filename.side_effect = lambda filename: open(filename, "w").write("model content")

        models = load_models("mock_dir", ["apparent_temperature_max"])
        self.assertIsInstance(models["apparent_temperature_max"], xgb.XGBRegressor)

    # Test: Reverse normalization function
    def test_reverse_normalization(self):
        result = reverse_normalization(0.5, 10, 20)
        self.assertEqual(result, 15)

    # Test: Daily weather predictions
    @patch("main.MIN_MAX_DAILY", {"apparent_temperature_max": {"min": -10, "max": 40}})
    @patch("main.reverse_normalization")
    def test_predict_daily_weather(self, mock_reverse_normalization):
        mock_reverse_normalization.side_effect = lambda val, min_val, max_val: val * (max_val - min_val) + min_val
        mock_models = {"apparent_temperature_max": MagicMock()}
        mock_models["apparent_temperature_max"].predict.return_value = [0.2, 0.4, 0.6, 0.8, 1.0]

        result = predict_daily_weather(mock_models)
        self.assertEqual(len(result), 7)
        self.assertIn("apparent_temperature_max", result.columns)

    # Test: Hourly weather predictions
    @patch("main.MIN_MAX_HOURLY", {"apparent_temperature": {"min": -10, "max": 40}})
    @patch("main.reverse_normalization")
    def test_predict_hourly_weather(self, mock_reverse_normalization):
        mock_reverse_normalization.side_effect = lambda val, min_val, max_val: val * (max_val - min_val) + min_val
        mock_models = {"apparent_temperature": MagicMock()}
        mock_models["apparent_temperature"].predict.return_value = [0.1] * 24

        date = datetime.now().date()
        result = predict_hourly_weather(mock_models, date)
        self.assertEqual(len(result), 24)
        self.assertIn("apparent_temperature", result.columns)

    # Test: Daily UI rendering
    @patch("main.predict_daily_weather")
    @patch("streamlit.date_input")
    @patch("streamlit.dataframe")
    @patch("streamlit.pyplot")
    def test_daily_ui(self, mock_pyplot, mock_dataframe, mock_date_input, mock_predict_daily_weather):
        mock_date_input.return_value = datetime.now().date()
        mock_predict_daily_weather.return_value = pd.DataFrame({
            "date": [datetime.now().date() + timedelta(days=i) for i in range(7)],
            "apparent_temperature_max": [30, 32, 31, 33, 34, 35, 36],
        })

        from main import main
        main()

        mock_date_input.assert_called()
        mock_dataframe.assert_called()
        mock_pyplot.assert_called()

    # Test: Hourly UI rendering
    @patch("main.predict_hourly_weather")
    @patch("streamlit.dataframe")
    @patch("streamlit.pyplot")
    @patch("streamlit.button")
    def test_hourly_ui(self, mock_button, mock_pyplot, mock_dataframe, mock_predict_hourly_weather):
        mock_button.return_value = True
        mock_predict_hourly_weather.return_value = pd.DataFrame({
            "hour": list(range(24)),
            "apparent_temperature": [30] * 24,
        })

        from main import main
        main()

        mock_button.assert_called()
        mock_dataframe.assert_called()
        mock_pyplot.assert_called()

    # Test: Handle no data in GCS
    @patch("google.cloud.storage.Client")
    def test_no_data(self, mock_client):
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Simulate empty file download
        mock_blob.download_to_filename.side_effect = lambda filename: open(filename, "w").write("")

        with self.assertRaises(Exception):
            get_min_max("mock_path")

    # Test: Invalid date input
    @patch("streamlit.date_input")
    def test_invalid_date(self, mock_date_input):
        mock_date_input.return_value = datetime.now().date() - timedelta(days=1)

        from main import main
        main()

        mock_date_input.assert_called()

    # Test: Model prediction failures
    @patch("main.predict_daily_weather")
    @patch("streamlit.error")
    def test_model_prediction_failures(self, mock_error, mock_predict_daily_weather):
        mock_predict_daily_weather.side_effect = Exception("Prediction failed")

        from main import main
        main()

        mock_error.assert_called_with("An error occurred during prediction: Prediction failed")


if __name__ == "__main__":
    unittest.main()
