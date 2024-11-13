from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import logging
from weather_data_collection import setup_session, fetch_daily_weather_data, process_daily_weather_data, save_data_to_csv, fetch_hourly_weather_data, process_hourly_weather_data, upload_to_gcs
from weather_data_visualization import generate_seasonal_trend_plots, generate_plots

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
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('weather_data_pipeline', default_args=default_args, schedule_interval=None)


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

# Set task dependencies
daily_weather_task >> hourly_weather_task >> preprocess_daily_task >> preprocess_hourly_task >> feature_engineering_task >> eda_and_visualizations_task >> generate_and_save_schema_stats_task >> validate_data_task >> schema_quality_test_task >> email_notification_task