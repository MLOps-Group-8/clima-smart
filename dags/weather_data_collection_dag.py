from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.email import send_email
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
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
from utils import read_data_from_gcs, save_data_to_gcs, save_object_to_gcs, load_object_from_gcs, save_plot_to_gcs
from constants import *

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
    'retry_delay': timedelta(minutes=2),
}

dag = DAG('weather_data_pipeline', default_args=default_args, 
          description = 'DAG to collect, preprocess, and analyze weather data', 
          schedule_interval=None,
          catchup=False)

# Define function to notify failure or sucess via an email
def notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to='keshiarun01@gmail.com',
        subject='Success Notification from Airflow',
        html_content='<p>The task succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def notify_failure(context):
    failure_email = EmailOperator(
        task_id='failure_email',
        to='keshiarun01@gmail.com',
        subject='Failure Notification from Airflow',
        html_content='<p>The task failed.</p>',
        dag=context['dag']
    )
    failure_email.execute(context=context)


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

def eda_and_visualizations():
    logging.info("Starting EDA and visualizations.")
    
    daily_data = read_data_from_gcs(BUCKET_NAME, ENGINEERED_DAILY_DATA_PATH)
    hourly_data = read_data_from_gcs(BUCKET_NAME, ENGINEERED_HOURLY_DATA_PATH)

    # Correlation heatmap for hourly data (numeric columns only)
    plt.figure(figsize=(20, 16))
    sns.heatmap(hourly_data.select_dtypes(include=[np.number]).corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap - Hourly Data')
    save_plot_to_gcs(BUCKET_NAME, 'correlation_heatmap_hourly')
    plt.clf()

    # Correlation heatmap for daily data (numeric columns only)
    plt.figure(figsize=(20, 16))
    sns.heatmap(daily_data.select_dtypes(include=[np.number]).corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap - Daily Data')
    save_plot_to_gcs(BUCKET_NAME, 'correlation_heatmap_daily')
    plt.clf()

    # Time series plot of temperature and precipitation (hourly data)
    # plt.figure(figsize=(20, 10))
    # plt.plot(hourly_data['datetime'], hourly_data['temperature_2m'], label='Temperature')
    # plt.plot(hourly_data['datetime'], hourly_data['precipitation'], label='Precipitation')
    # plt.title('Temperature and Precipitation Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.legend()
    # save_plot_to_gcs(BUCKET_NAME, 'time_series_temp_precip')
    # plt.clf()

    # Distribution of temperature (daily data)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=daily_data, x='temperature_2m_max', kde=True)
    plt.title('Distribution of Daily Maximum Temperature')
    save_plot_to_gcs(BUCKET_NAME, 'distribution_daily_max_temp')
    plt.clf()

    # Box plot of precipitation by season (daily data)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=daily_data, x='season', y='precipitation_sum')
    plt.title('Precipitation by Season')
    save_plot_to_gcs(BUCKET_NAME, 'boxplot_precip_by_season')
    plt.clf()
    
    logging.info("EDA and visualizations completed.")

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
    on_failure_callback=notify_failure,
    dag=dag
)

hourly_weather_task = PythonOperator(
    task_id='fetch_and_save_hourly_weather',
    python_callable=get_hourly_weather_data,
    on_failure_callback=notify_failure,
    dag=dag
)

preprocess_daily_task = PythonOperator(
    task_id='preprocess_daily_data',
    python_callable=preprocess_daily_weather_data,
    on_failure_callback=notify_failure,
    dag=dag
)

preprocess_hourly_task = PythonOperator(
    task_id='preprocess_hourly_data',
    python_callable=preprocess_hourly_weather_data,
    on_failure_callback=notify_failure,
    dag=dag
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=perform_feature_engineering,
    on_failure_callback=notify_failure,
    dag=dag
)

eda_and_visualizations_task = PythonOperator(
    task_id='eda_and_visualization',
    python_callable=eda_and_visualizations,
    on_failure_callback=notify_failure,
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
    on_failure_callback=notify_failure,
    dag=dag
)

# Validation tasks
validate_data_task = PythonOperator(
    task_id='validate_weather_data',
    python_callable=validate_daily_and_houlry_weather_data,
    on_failure_callback=notify_failure,
    dag=dag
)

schema_quality_test_task = PythonOperator(
    task_id='test_data_quality_and_schema',
    python_callable=test_weather_data_quality_and_schema,
    on_failure_callback=notify_failure,
    dag=dag
)

email_notification_task = EmailOperator(
    task_id='send_email_notification',
    to='keshiarun01@gmail.com',
    subject='Dag Completed Successfully',
    html_content='<p>Dag Completed</p>',
    dag=dag,
)

# Task to trigger the ModelPipeline DAG
trigger_model_pipeline_task = TriggerDagRunOperator(
    task_id='trigger_model_pipeline_task',
    trigger_dag_id='ModelDevelopmentPipeline1',
    trigger_rule=TriggerRule.ALL_DONE,  # Ensure this task runs only if all upstream tasks succeed
    dag=dag,
)

# Set task dependencies
daily_weather_task >> hourly_weather_task >> preprocess_daily_task >> preprocess_hourly_task >> feature_engineering_task >> eda_and_visualizations_task >> generate_and_save_schema_stats_task >> validate_data_task >> schema_quality_test_task >> email_notification_task >> trigger_model_pipeline_task

