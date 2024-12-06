import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import xgboost as xgb
import tempfile

# Mocking Streamlit components
import streamlit as st

class TestWeatherForecastApp(unittest.TestCase):

    @patch('streamlit.sidebar.text_input')
    @patch('streamlit.sidebar.info')
    @patch('streamlit.sidebar.success')
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    @patch('streamlit.table')
    @patch('streamlit.line_chart')
    @patch('google.cloud.storage.Client')
    def test_fetch_data_from_gcs(self, mock_client, mock_line_chart, mock_table, mock_dataframe,
                                  mock_subheader, mock_success, mock_info, mock_text_input):
        """Test if data fetching from GCS works correctly."""
        # Mock the storage client
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Mock file download
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"temperature_2m_max,temperature_2m_min,apparent_temperature_min,rain_sum,showers_sum,daylight_duration\n20,10,15,2,1,10")
        temp_file.close()
        mock_blob.download_to_filename.side_effect = lambda filename: open(filename, 'wb').write(
            open(temp_file.name, 'rb').read()
        )

        # Call the function
        from main import fetch_data_from_gcs
        data = fetch_data_from_gcs("mock_bucket", "mock_path")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 1)  # Check if the data has the expected number of rows

    @patch('google.cloud.storage.Client')
    def test_load_model_from_gcs(self, mock_client):
        """Test if model loading from GCS works correctly."""
        # Mock the storage client
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Mock file download
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"xgboost model binary content")
        temp_file.close()
        mock_blob.download_to_filename.side_effect = lambda filename: open(filename, 'wb').write(
            open(temp_file.name, 'rb').read()
        )

        # Call the function
        from main import load_model_from_gcs
        model = load_model_from_gcs("mock_bucket", "mock_model_path")
        self.assertIsInstance(model, xgb.Booster)

    @patch('streamlit.sidebar.text_input')
    @patch('streamlit.sidebar.info')
    @patch('streamlit.sidebar.success')
    @patch('streamlit.dataframe')
    @patch('streamlit.table')
    @patch('streamlit.line_chart')
    @patch('google.cloud.storage.Client')
    def test_app_run(self, mock_client, mock_line_chart, mock_table, mock_dataframe,
                     mock_success, mock_info, mock_text_input):
        """Test the app's execution."""
        # Mock Streamlit inputs
        mock_text_input.side_effect = ["mock_model_path", "mock_csv_path"]

        # Mock GCS data fetching
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"temperature_2m_max,temperature_2m_min,apparent_temperature_min,rain_sum,showers_sum,daylight_duration\n20,10,15,2,1,10")
        temp_file.close()
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_to_filename.side_effect = lambda filename: open(filename, 'wb').write(
            open(temp_file.name, 'rb').read()
        )

        # Mock GCS model fetching
        model_temp_file = tempfile.NamedTemporaryFile(delete=False)
        model_temp_file.write(b"xgboost model binary content")
        model_temp_file.close()
        mock_blob.download_to_filename.side_effect = lambda filename: open(filename, 'wb').write(
            open(model_temp_file.name, 'rb').read()
        )

        # Mock the XGBoost model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([30.0, 32.0, 31.5])

        with patch('main.load_model_from_gcs', return_value=mock_model):
            from main import main
            main()

        # Assertions to ensure correct outputs
        mock_dataframe.assert_called()  # Ensure data display is called
        mock_table.assert_called()      # Ensure table with predictions is shown
        mock_line_chart.assert_called() # Ensure chart with predictions is plotted

if __name__ == '__main__':
    unittest.main()
