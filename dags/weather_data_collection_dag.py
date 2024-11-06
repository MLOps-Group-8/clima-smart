from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import logging
from weather_data_collection import (
    setup_session,
    fetch_daily_weather_data,
    process_daily_weather_data,
    save_data_to_csv,
    fetch_hourly_weather_data,
    process_hourly_weather_data,
    upload_to_gcs
)
from weather_data_preprocessing import preprocess_daily_data, preprocess_hourly_data
from feature_engineering import feature_engineering
from weather_data_validation import validate_weather_data, test_data_quality_and_schema, save_schema_and_stats

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
    save_data_to_csv(daily_data, 'daily_weather_data.csv')
    upload_to_gcs(BUCKET_NAME, 'daily_weather_data.csv', 'weather_data/daily_weather_data.csv')
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
    save_data_to_csv(hourly_data, 'hourly_weather_data.csv')
    upload_to_gcs(BUCKET_NAME, 'hourly_weather_data.csv', 'weather_data/hourly_weather_data.csv')
    logging.info("Hourly weather data task completed.")

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
    python_callable=preprocess_daily_data,
    dag=dag
)

preprocess_hourly_task = PythonOperator(
    task_id='preprocess_hourly_data',
    python_callable=preprocess_hourly_data,
    dag=dag
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering_and_visualization',
    python_callable=feature_engineering,
    dag=dag
)

# Updated task for generating and saving schema and stats
generate_and_save_schema_stats = PythonOperator(
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
    python_callable=validate_weather_data,
    dag=dag
)

schema_quality_test_task = PythonOperator(
    task_id='test_data_quality_and_schema',
    python_callable=test_data_quality_and_schema,
    dag=dag
)

# Set task dependencies
daily_weather_task >> hourly_weather_task >> preprocess_daily_task >> preprocess_hourly_task >> feature_engineering_task >> generate_and_save_schema_stats >> validate_data_task >> schema_quality_test_task
