from unittest import TestCase
from dags.weather_data_preprocessing import preprocess_daily_data, preprocess_hourly_data
import pandas as pd
import numpy as np

class TestWeatherDataPreprocessing(TestCase):

    def test_preprocess_daily_data(self):
        data = pd.DataFrame({'date': ['2023-01-01'], 'temperature_2m_max': [25]})
        result = preprocess_daily_data(data)
        self.assertIn('is_weekend', result.columns)

    def test_preprocess_hourly_data(self):
        hourly_data = pd.DataFrame({
            'datetime': pd.date_range(start='2023-01-01', periods=24, freq='H'),
            'temperature_2m': [np.nan, 20] + [25] * 22,
            'precipitation': [0] * 24,
        })
        result = preprocess_hourly_data(hourly_data)
        self.assertTrue(result['temperature_2m'].isnull().sum() == 0)
        self.assertIn('is_precipitation', result.columns)