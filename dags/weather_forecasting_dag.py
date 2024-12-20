from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG('weather_forecasting_pipeline', default_args=default_args, 
          description = 'DAG to collect, preprocess, analyze and develope models for weather data', 
          schedule_interval='@daily',
          tags=['weather', 'forecasting'],
          catchup=False)

start_task = BashOperator(
    task_id="start_task",
    bash_command="echo 'Starting the weather forecasting pipelines'",
    dag=dag,
  )

# Trigger daily data collection DAG
daily_weather_data_forecasting_trigger_task = TriggerDagRunOperator(
    task_id='daily_weather_data_forecasting_trigger_task',
    trigger_dag_id='daily_weather_data_pipeline', 
    dag=dag,
)

# Trigger hourly tasks
hourly_weather_data_forecasting_trigger_task = TriggerDagRunOperator(
    task_id='hourly_weather_data_forecasting_trigger_task',
    trigger_dag_id='hourly_weather_data_pipeline', 
    dag=dag,
)

# Final notification
email_notification_task = EmailOperator(
    task_id='email_notification_task',
    to='keshiarun01@gmail.com', 
    subject='Weather Forecasting Pipeline Completed',
    html_content='<p>The Weather Forecasting Pipeline has completed successfully.</p>',
    dag=dag,
)

# Define dependencies
start_task >> daily_weather_data_forecasting_trigger_task
daily_weather_data_forecasting_trigger_task >> hourly_weather_data_forecasting_trigger_task
hourly_weather_data_forecasting_trigger_task >> email_notification_task
