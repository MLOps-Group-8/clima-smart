from unittest import TestCase, mock
from dags.weather_data_collection import setup_session, fetch_daily_weather_data, process_daily_weather_data, process_hourly_weather_data
import pandas as pd

class TestWeatherDataCollection(TestCase):

    @mock.patch('weather_data_collection.requests_cache.CachedSession')
    def test_setup_session(self, mock_session):
        client = setup_session()
        self.assertIsNotNone(client)

    @mock.patch('weather_data_collection.openmeteo_requests.Client')
    def test_fetch_daily_weather_data(self, mock_client):
        mock_response = {'daily': {}}
        mock_client.weather_api.return_value = mock_response
        client = setup_session()
        data = fetch_daily_weather_data(client, 'http://api.test', {})
        self.assertIsInstance(data, dict)
    
    @mock.patch('weather_data_collection.openmeteo_requests.Client')
    def test_process_daily_weather_data(self, mock_client):
        mock_response = mock.MagicMock()
        mock_client.weather_api.return_value = mock_response
        result = process_daily_weather_data(mock_response)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('date', result.columns)

    @mock.patch('weather_data_collection.openmeteo_requests.Client')
    def test_process_hourly_weather_data(self, mock_client):
        mock_response = mock.MagicMock()
        mock_client.weather_api.return_value = mock_response
        result = process_hourly_weather_data(mock_response)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('datetime', result.columns)
