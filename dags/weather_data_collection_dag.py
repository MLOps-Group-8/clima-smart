from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import logging
from weather_data_collection import (
    setup_session,
    fetch_daily_weather_data,
    process_daily_weather_data,
    fetch_hourly_weather_data,
    process_hourly_weather_data
)
from weather_data_preprocessing import preprocess_daily_data, preprocess_hourly_data
from feature_engineering import feature_engineering
from weather_data_validation import validate_weather_data, test_data_quality_and_schema
from utils import read_data_from_gcs, save_data_to_gcs, save_object_to_gcs, load_object_from_gcs
from constants import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = "https://archive-api.open-meteo.com/v1/archive"
BUCKET_NAME = 'us-east1-climasmart-fefe9cc2-bucket'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG('weather_data_pipeline', default_args=default_args, schedule_interval=None)

# Task to fetch and save daily weather data
def get_daily_weather_data():
    logging.info("Starting the daily weather data task.")
    client = setup_session()
    params = {
        "latitude": 42.36,
        "longitude": -71.057,
        "timezone": "America/New_York",
        "start_date": "2000-01-01",
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "temperature_unit": "fahrenheit"
    }
    response = fetch_daily_weather_data(client, API_URL, params)
    daily_data = process_daily_weather_data(response)
    save_data_to_gcs(daily_data, BUCKET_NAME, DAILY_DATA_PATH)
    logging.info("Daily weather data task completed.")

# Task to fetch and save hourly weather data
def get_hourly_weather_data():
    logging.info("Starting the hourly weather data task.")
    client = setup_session()
    params = {
        "latitude": 42.36,
        "longitude": -71.057,
        "timezone": "America/New_York",
        "start_date": "2020-01-01",
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "temperature_unit": "fahrenheit"
    }
    response = fetch_hourly_weather_data(client, API_URL, params)
    hourly_data = process_hourly_weather_data(response)
    save_data_to_gcs(hourly_data, BUCKET_NAME, HOURLY_DATA_PATH)
    logging.info("Hourly weather data task completed.")

# Task to preprocess daily data
def preprocess_daily_weather_data():
    logging.info("Starting the daily data preprocessing task.")
    daily_data = read_data_from_gcs(BUCKET_NAME, DAILY_DATA_PATH)
    df = preprocess_daily_data(daily_data)
    save_data_to_gcs(df, BUCKET_NAME, PREPROCESSED_DAILY_DATA_PATH)
    logging.info("Daily data preprocessing task completed.")

# Task to preprocess hourly data
def preprocess_hourly_weather_data():
    logging.info("Starting the hourly data preprocessing task.")
    hourly_data = read_data_from_gcs(BUCKET_NAME, HOURLY_DATA_PATH)
    df = preprocess_hourly_data(hourly_data)
    save_data_to_gcs(df, BUCKET_NAME, PREPROCESSED_HOURLY_DATA_PATH)
    logging.info("Hourly data preprocessing task completed.")

# Task to perform feature engineering and visualization
def perform_feature_engineering():
    logging.info("Starting the feature engineering and visualization task.")
    
    daily_data = read_data_from_gcs(BUCKET_NAME, PREPROCESSED_DAILY_DATA_PATH)
    hourly_data = read_data_from_gcs(BUCKET_NAME, PREPROCESSED_HOURLY_DATA_PATH)
    
    hourly_data, daily_data = feature_engineering(hourly_data, daily_data)
    
    save_data_to_gcs(hourly_data, BUCKET_NAME, ENGINEERED_HOURLY_DATA_PATH)
    save_data_to_gcs(daily_data, BUCKET_NAME, ENGINEERED_DAILY_DATA_PATH)
    logging.info("Feature engineering and visualization task completed.")

# Task to generate and save schema and stats
def save_schema_and_stats(daily_schema, hourly_schema, daily_stats, hourly_stats):
    """Save schemas and statistics to GCS for future reference."""
    logging.info("Starting the schema and stats saving task.")
    save_object_to_gcs(BUCKET_NAME, daily_schema, DAILY_SCHEMA_PATH)
    save_object_to_gcs(BUCKET_NAME, hourly_schema, HOURLY_SCHEMA_PATH)
    save_object_to_gcs(BUCKET_NAME, daily_stats, DAILY_STATS_PATH)
    save_object_to_gcs(BUCKET_NAME, hourly_stats, HOURLY_STATS_PATH)
    logging.info("Schema and stats saved to GCS.")

# Task to validate the weather data
def validate_daily_and_houlry_weather_data():
    logging.info("Starting the data validation task.")
    
    daily_data = read_data_from_gcs(BUCKET_NAME, ENGINEERED_DAILY_DATA_PATH)
    hourly_data = read_data_from_gcs(BUCKET_NAME, ENGINEERED_HOURLY_DATA_PATH)
    validate_weather_data(daily_data, hourly_data)

    logging.info("Data validation task completed.")

# Task to test data quality and schema
def test_weather_data_quality_and_schema():
    logging.info("Starting the data quality and schema test task.")
    daily_schema = load_object_from_gcs(BUCKET_NAME, DAILY_SCHEMA_PATH)
    hourly_schema = load_object_from_gcs(BUCKET_NAME, HOURLY_SCHEMA_PATH)
    test_data_quality_and_schema(daily_schema, hourly_schema)
    logging.info("Data quality and schema test task completed.")

# Define Airflow tasks
daily_weather_task = PythonOperator(
    task_id='fetch_and_save_daily_weather',
    python_callable=get_daily_weather_data,
    dag=dag
)

hourly_weather_task = PythonOperator(
    task_id='fetch_and_save_hourly_weather',
    python_callable=get_hourly_weather_data,
    dag=dag
)

preprocess_daily_task = PythonOperator(
    task_id='preprocess_daily_data',
    python_callable=preprocess_daily_weather_data,
    dag=dag
)

preprocess_hourly_task = PythonOperator(
    task_id='preprocess_hourly_data',
    python_callable=preprocess_hourly_weather_data,
    dag=dag
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering_and_visualization',
    python_callable=perform_feature_engineering,
    dag=dag
)

# Updated task for generating and saving schema and stats
generate_and_save_schema_stats_task = PythonOperator(
    task_id='generate_and_save_schema_stats',
    python_callable=save_schema_and_stats,
    op_args=['daily_schema.pkl', 'hourly_schema.pkl', 'daily_stats.pkl', 'hourly_stats.pkl'],
    op_kwargs={
        'bucket_name': BUCKET_NAME,
        'destination_path': 'weather_data_validation'
    },
    dag=dag
)

# Validation tasks
validate_data_task = PythonOperator(
    task_id='validate_weather_data',
    python_callable=validate_daily_and_houlry_weather_data,
    dag=dag
)

schema_quality_test_task = PythonOperator(
    task_id='test_data_quality_and_schema',
    python_callable=test_weather_data_quality_and_schema,
    dag=dag
)

# Set task dependencies
daily_weather_task >> hourly_weather_task >> preprocess_daily_task >> preprocess_hourly_task >> feature_engineering_task >> generate_and_save_schema_stats_task >> validate_data_task >> schema_quality_test_task
