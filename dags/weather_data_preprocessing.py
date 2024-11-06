# weather_data_preprocessing.py

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from google.cloud import storage
import pandas as pd
import numpy as np
import logging
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


BUCKET_NAME = 'us-east1-climasmart-fefe9cc2-bucket'

def load_data_from_gcs(bucket_name, file_name):
    """Load CSV data from Google Cloud Storage into a DataFrame."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_string()
    df = pd.read_csv(io.BytesIO(data))
    logging.info(f"Loaded {file_name} from GCS.")
    return df

def save_data_to_gcs(df, bucket_name, file_name):
    """Save DataFrame as a CSV file in Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    blob.upload_from_file(output, content_type='text/csv')
    logging.info(f"Saved {file_name} to GCS.")

def preprocess_daily_data():
    df = load_data_from_gcs(BUCKET_NAME, 'weather_data/daily_weather_data.csv')

    # Preprocessing steps for daily data
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    columns_to_ffill = [
        'weather_code', 'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_max',
        'apparent_temperature_min', 'sunshine_duration', 'precipitation_sum', 'rain_sum',
        'snowfall_sum', 'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant',
        'shortwave_radiation_sum', 'et0_fao_evapotranspiration'
    ]
    df[columns_to_ffill] = df[columns_to_ffill].ffill()
    df.drop(['uv_index_max', 'uv_index_clear_sky_max', 'precipitation_probability_max', 'sunrise', 'sunset'], axis=1, inplace=True)

    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df[column] = df[column].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    for column in df.select_dtypes(include=[np.number]).columns:
        remove_outliers(df, column)

    df['season'] = pd.cut(df.index.month, bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'], include_lowest=True)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = ((df[column] - df[column].min()) / (df[column].max() - df[column].min())).round(3)

    df['weather_code'] = df['weather_code'].astype('category')
    df['is_weekend'] = df['is_weekend'].astype('category')

    df.reset_index(inplace=True)

    save_data_to_gcs(df, BUCKET_NAME, 'weather_data/preprocessed_daily_data.csv')

def preprocess_hourly_data():
    df = load_data_from_gcs(BUCKET_NAME, 'weather_data/hourly_weather_data.csv')

    # Preprocessing steps for hourly data
    df['date'] = pd.to_datetime(df['datetime'])
    df.set_index('date', inplace=True)

    def handle_missing_values(df, column):
        if df[column].isnull().sum() > 0:
            if column in ['precipitation', 'rain', 'showers', 'snowfall']:
                df[column].fillna(0, inplace=True)
            elif 'temperature' in column or 'humidity' in column or 'pressure' in column:
                df[column].interpolate(method='time', inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)

    for column in df.columns:
        handle_missing_values(df, column)

    df['weather_code'] = df['weather_code'].astype('category')
    df['is_precipitation'] = ((df['precipitation'] > 0) | (df['rain'] > 0) | (df['showers'] > 0) | (df['snowfall'] > 0)).astype(int)
    df['season'] = pd.cut(df.index.month, bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'], include_lowest=True)
    df['hour'] = df.index.hour
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    for column in df.select_dtypes(include=[np.number]).columns:
        min_val, max_val = df[column].min(), df[column].max()
        df[column] = ((df[column] - min_val) / (max_val - min_val)).round(3)

    df.dropna(axis=1, how='all', inplace=True)
    df['is_weekend'] = df['is_weekend'].astype('category')
    df['is_precipitation'] = df['is_precipitation'].astype('category')

    save_data_to_gcs(df, BUCKET_NAME, 'weather_data/preprocessed_hourly_data.csv')
