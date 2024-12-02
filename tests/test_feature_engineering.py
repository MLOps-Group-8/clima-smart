import unittest
import pandas as pd
import numpy as np
from dags.feature_engineering import engineer_hourly_features, engineer_daily_features, daily_feature_engineering, hourly_feature_engineering

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Sample hourly and daily data
        self.hourly_data = pd.DataFrame({
            'datetime': pd.date_range(start='2023-01-01', periods=48, freq='H'),
            'season': ['Winter'] * 48,
            'temperature_2m': np.random.uniform(-5, 15, 48),
            'precipitation': np.random.uniform(0, 10, 48),
            'snowfall': np.random.uniform(0, 5, 48),
            'wind_speed_10m': np.random.uniform(0, 20, 48),
            'relative_humidity_2m': np.random.uniform(30, 80, 48),
            'dew_point_2m': np.random.uniform(-10, 10, 48),
        })

        self.daily_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'season': ['Winter'] * 10,
            'temperature_2m_max': np.random.uniform(-5, 15, 10),
            'temperature_2m_min': np.random.uniform(-15, 5, 10),
            'precipitation_sum': np.random.uniform(0, 20, 10),
            'precipitation_hours': np.random.randint(0, 24, 10),
            'sunshine_duration': np.random.uniform(0, 12, 10),
            'daylight_duration': [12] * 10,
            'rain_sum': np.random.uniform(0, 10, 10)
        })

    def test_engineer_hourly_features(self):
        result = engineer_hourly_features(self.hourly_data)
        self.assertIn('is_freezing', result.columns)
        self.assertIn('heat_index', result.columns)
        self.assertIn('wind_category', result.columns)
        self.assertTrue(all(result['is_freezing'].isin([0, 1])))

    def test_engineer_daily_features(self):
        result = engineer_daily_features(self.daily_data)
        self.assertIn('temperature_range', result.columns)
        self.assertIn('is_hot_day', result.columns)
        self.assertIn('rain_ratio', result.columns)
        self.assertTrue(all(result['is_hot_day'].isin([0, 1])))
    
    def test_daily_feature_engineering(self):
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'season': ['Winter'] * 5,
            'temperature_2m_max': np.random.uniform(-10, 15, 5),
            'temperature_2m_min': np.random.uniform(-20, 5, 5)
        })
        result = daily_feature_engineering(data)
        self.assertIn('temperature_range', result.columns)
        self.assertTrue(result['temperature_2m_max'].isnull().sum() == 0)

    def test_hourly_feature_engineering(self):
        data = pd.DataFrame({
            'datetime': pd.date_range(start='2023-01-01', periods=48, freq='H'),
            'temperature_2m': np.random.uniform(-5, 10, 48),
            'precipitation': np.random.uniform(0, 20, 48)
        })
        result = hourly_feature_engineering(data)
        self.assertIn('wind_chill', result.columns)
        self.assertIn('precip_rolling_sum_24h', result.columns)
