from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
import os
import pandas as pd
from hourly_model_training_v2 import train_and_save_models, monitor_model_performance, load_models
import logging
from constants import *
from utils import upload_to_gcs
from hourly_model_analysis import bias_analysis, sensitivity_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    dag_id='hourly_model_development_pipeline',
    default_args=default_args,
    description='Pipeline for training, monitoring, and retraining models for hourly weather data',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['weather', 'forecasting', 'hourly'],
)

# Configuration
LOCAL_HOURLY_FILE = '/tmp/weather_data_engineered_hourly_data.csv'
MODEL_DIR = '/tmp/models/hourly'
TARGET_FEATURES = ['apparent_temperature', 'precipitation', 'rain']
DATE_FEATURES = ['hour', 'month', 'day_of_year', 'week_of_year', 'is_weekend']
ANALYSIS_DIR = '/tmp/analysis/hourly'
METRIC_THRESHOLDS = {'rmse': 5.0, 'r2': 0.8}

def download_hourly_data():
    """Download hourly training data from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(ENGINEERED_HOURLY_DATA_PATH)
    blob.download_to_filename(LOCAL_HOURLY_FILE)

    data = pd.read_csv(LOCAL_HOURLY_FILE)
    data['date'] = pd.to_datetime(data['datetime'])
    data['hour'] = data['date'].dt.hour
    data['month'] = data['date'].dt.month
    data['day_of_year'] = data['date'].dt.day_of_year
    data['week_of_year'] = data['date'].dt.isocalendar().week
    data['is_weekend'] = data['date'].dt.weekday >= 5
    data.to_csv(LOCAL_HOURLY_FILE, index=False)

def train_hourly_models():
    """Train models for hourly data."""
    train_and_save_models(
        data_path=LOCAL_HOURLY_FILE,
        model_dir=MODEL_DIR,
        target_features=TARGET_FEATURES
    )

def upload_hourly_models():
    """Upload trained hourly models to GCS."""
    storage_client = storage.Client()
    for target in TARGET_FEATURES:
        model_path = os.path.join(MODEL_DIR, f"{target}_model.json")
        remote_path = f"models/hourly/{target}_model.json"
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(model_path)
        logging.info(f"Uploaded {model_path} to {remote_path}")

def perform_bias_analysis():
    """Perform bias analysis for hourly data and upload results to GCS."""
    data = pd.read_csv(LOCAL_HOURLY_FILE)
    models = load_models(MODEL_DIR, TARGET_FEATURES)
    bias_results = bias_analysis(models, data, DATE_FEATURES, TARGET_FEATURES, ANALYSIS_DIR)

    # Upload bias plots to GCS
    for target in TARGET_FEATURES:
        plot_path = os.path.join(ANALYSIS_DIR, f"{target}_bias_plot.png")
        upload_to_gcs(BUCKET_NAME, plot_path, f"analysis/hourly/bias/{target}_bias_plot.png")

def perform_sensitivity_analysis():
    """Perform sensitivity analysis for hourly data and upload results to GCS."""
    data = pd.read_csv(LOCAL_HOURLY_FILE)
    models = load_models(MODEL_DIR, TARGET_FEATURES)
    sensitivity_results = sensitivity_analysis(models, data, DATE_FEATURES, TARGET_FEATURES, ANALYSIS_DIR)

    # Upload sensitivity results to GCS
    for target in TARGET_FEATURES:
        csv_path = os.path.join(ANALYSIS_DIR, f"{target}_sensitivity.csv")
        upload_to_gcs(BUCKET_NAME, csv_path, f"analysis/hourly/sensitivity/{target}_sensitivity.csv")


def monitor_hourly_models():
    """Monitor the performance of hourly models."""
    models = load_models(MODEL_DIR, TARGET_FEATURES)
    performance_metrics, retrain_needed = monitor_model_performance(
        models=models,
        data_path=LOCAL_HOURLY_FILE,
        target_features=TARGET_FEATURES,
        thresholds=METRIC_THRESHOLDS
    )
    logging.info(f"Hourly model performance: {performance_metrics}")
    if retrain_needed:
        logging.info("Retraining triggered due to performance degradation.")
        train_hourly_models()
        upload_hourly_models()

# Define tasks
download_hourly_data_task = PythonOperator(
    task_id='download_hourly_data',
    python_callable=download_hourly_data,
    dag=dag,
)

train_hourly_models_task = PythonOperator(
    task_id='train_hourly_models',
    python_callable=train_hourly_models,
    dag=dag,
)

bias_analysis_task = PythonOperator(
    task_id='perform_hourly_bias_analysis',
    python_callable=perform_bias_analysis,
)

sensitivity_analysis_task = PythonOperator(
    task_id='perform_hourly_sensitivity_analysis',
    python_callable=perform_sensitivity_analysis,
)

upload_hourly_models_task = PythonOperator(
    task_id='upload_hourly_models',
    python_callable=upload_hourly_models,
    dag=dag,
)

monitor_hourly_models_task = PythonOperator(
    task_id='monitor_hourly_models',
    python_callable=monitor_hourly_models,
    dag=dag,
)

# Define task dependencies
download_hourly_data_task >> train_hourly_models_task >> [bias_analysis_task, sensitivity_analysis_task] >> upload_hourly_models_task >> monitor_hourly_models_task
