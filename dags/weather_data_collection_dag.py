from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import logging
from weather_data_collection import setup_session, fetch_daily_weather_data, process_daily_weather_data, save_data_to_csv, fetch_hourly_weather_data, process_hourly_weather_data, upload_to_gcs
from weather_data_visualization import generate_seasonal_trend_plots, generate_plots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


API_URL = "https://archive-api.open-meteo.com/v1/archive"
BUCKET_NAME = 'clima-smart-data-collection'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('weather_data_pipeline', default_args=default_args, schedule_interval=None)

def get_daily_weather_data():
    logging.info("Starting the daily weather data task.")
    client = setup_session()
    params = {"latitude": 42.36, "longitude": -71.057, "timezone": "America/New_York", "start_date": "2000-01-01", "end_date": "2024-09-30", "temperature_unit": "fahrenheit"}
    response = fetch_daily_weather_data(client, API_URL, params)
    daily_data = process_daily_weather_data(response)
    save_data_to_csv(daily_data, 'daily_weather_data.csv')
    upload_to_gcs(BUCKET_NAME, 'daily_weather_data.csv', 'weather_data/daily_weather_data.csv')
    logging.info("Daily weather data task completed.")
    
def get_hourly_weather_data():
    logging.info("Starting the hourly weather data task.")
    client = setup_session()
    params = {"latitude": 42.36, "longitude": -71.057, "timezone": "America/New_York", "start_date": "2020-01-01", "end_date": "2024-09-30", "temperature_unit": "fahrenheit"}
    response = fetch_hourly_weather_data(client, API_URL, params)
    hourly_data = process_hourly_weather_data(response)
    save_data_to_csv(hourly_data, 'hourly_weather_data.csv')
    upload_to_gcs(BUCKET_NAME, 'hourly_weather_data.csv', 'weather_data/hourly_weather_data.csv')
    logging.info("Hourly weather data task completed.")
    
def visualize_and_upload():
    logging.info("Starting the visualization weather data task.")
    generate_plots(BUCKET_NAME, 'weather_data/daily_weather_data.csv', 'weather_data/hourly_weather_data.csv', 'weather_data_plots')
    generate_seasonal_trend_plots(BUCKET_NAME, 'weather_data/daily_weather_data.csv', 'weather_data_plots')
    logging.info("Visualization task completed including uploading plots.")
    
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

visualize_and_upload_task = PythonOperator(
    task_id='visualize_and_upload',
    python_callable=visualize_and_upload,
    dag=dag
)

daily_weather_task >> hourly_weather_task >> visualize_and_upload_task
