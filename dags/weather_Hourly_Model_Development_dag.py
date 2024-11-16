from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
import logging
import os
import pickle
import pandas as pd
import numpy as np
from hourlymodeltraining import (
    load_data_from_gcs,
    process_data,
    train_model,
    save_model_to_gcs
)
from hourlymodelvalidation import evaluate_and_visualize_models
from hourlybiasdetection import calculate_metrics_for_features
from hourlymodelsensitivity import analyze_hourly_model_sensitivity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
# GCS bucket and file paths
BUCKET_NAME = 'clima-smart-data-collection'
ENGINEERED_HOURLY_DATA_PATH = 'weather_data/engineered_hourly_data.csv'
HOURLY_DATA_PLOTS_PATH = 'hourly_model_validation_plots/'

# Default args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Define the DAG
dag = DAG(
    'HourlyWeatherModelPipeline',
    default_args=default_args,
    description='DAG for training, validating, and analyzing hourly weather models',
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=False
)

# Temporary directory for saving data and models
TEMP_DIR = "/tmp/airflow_hourly_model_pipeline"
os.makedirs(TEMP_DIR, exist_ok=True)

# Email notification functions
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

# Task to load data from GCS
def load_data_task(**kwargs):
    logging.info("Loading data from GCS")
    data = load_data_from_gcs(BUCKET_NAME, ENGINEERED_HOURLY_DATA_PATH)
    data_path = os.path.join(TEMP_DIR, "raw_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    kwargs['ti'].xcom_push(key='raw_data_path', value=data_path)
    logging.info("Data loading task completed.")

# Task to process data
def process_data_task(**kwargs):
    logging.info("Processing data")
    raw_data_path = kwargs['ti'].xcom_pull(key='raw_data_path', task_ids='load_hourly_data')
    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)

    # Define features and targets
    features = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
        'cloud_cover', 'pressure_msl', 'wind_speed_10m', 'wind_direction_10m',
        'is_day', 'hour', 'is_weekend', 'month', 'is_holiday'
    ]
    targets = ['snowfall', 'rain', 'apparent_temperature', 'wind_gusts_10m']

    processed_data = process_data(raw_data, features, targets)

    # Save processed data for each target
    for target, splits in processed_data.items():
        for split, data_split in splits.items():
            split_path = os.path.join(TEMP_DIR, f"{target}_{split}.pkl")
            with open(split_path, "wb") as f:
                pickle.dump(data_split, f)
            kwargs['ti'].xcom_push(key=f"{target}_{split}_path", value=split_path)
    logging.info("Data processing task completed.")

# Task to train models
def train_model_task(**kwargs):
    logging.info("Training models for all targets")
    targets = ['snowfall', 'rain', 'apparent_temperature', 'wind_gusts_10m']

    for target in targets:
        # Pull data paths from XCom
        X_train_path = kwargs['ti'].xcom_pull(key=f"{target}_X_train_path", task_ids='process_hourly_data')
        X_val_path = kwargs['ti'].xcom_pull(key=f"{target}_X_val_path", task_ids='process_hourly_data')
        y_train_path = kwargs['ti'].xcom_pull(key=f"{target}_y_train_path", task_ids='process_hourly_data')
        y_val_path = kwargs['ti'].xcom_pull(key=f"{target}_y_val_path", task_ids='process_hourly_data')

        with open(X_train_path, "rb") as f:
            X_train = pickle.load(f)
        with open(X_val_path, "rb") as f:
            X_val = pickle.load(f)
        with open(y_train_path, "rb") as f:
            y_train = pickle.load(f)
        with open(y_val_path, "rb") as f:
            y_val = pickle.load(f)

        model = train_model(X_train, X_val, y_train, y_val)

        # Save the model
        model_path = os.path.join(TEMP_DIR, f"{target}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        kwargs['ti'].xcom_push(key=f"{target}_model_path", value=model_path)

    logging.info("Model training completed.")

# Task to validate models
def validate_models_task(**kwargs):
    logging.info("Validating models and generating visualizations")
    targets = ['snowfall', 'rain', 'apparent_temperature', 'wind_gusts_10m']
    models = {}
    processed_data = {}

    # Retrieve models and processed data from XCom
    for target in targets:
        # Model
        model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_hourly_models')
        with open(model_path, "rb") as f:
            models[target] = pickle.load(f)

        # Processed data
        processed_data[target] = {}
        for split in ['X_val', 'X_test', 'y_val', 'y_test', 'y_train']:
            split_path = kwargs['ti'].xcom_pull(key=f"{target}_{split}_path", task_ids='process_hourly_data')
            with open(split_path, "rb") as f:
                processed_data[target][split] = pickle.load(f)

    # Evaluate and visualize models
    evaluate_and_visualize_models(
        models=models,
        processed_data=processed_data,
        targets=targets,
        bucket_name=BUCKET_NAME,
        HOURLY_DATA_PLOTS_PATH=HOURLY_DATA_PLOTS_PATH
    )
    logging.info("Model validation and visualization task completed.")

def bias_detection_task(**kwargs):
    logging.info("Starting bias detection task")

    slicing_features = ['temperature_2m', 'precipitation']  # Features to slice by
    targets = ['snowfall', 'rain', 'apparent_temperature', 'wind_gusts_10m']
    
    # Attempt to retrieve necessary components for each target
    for target in targets:
        try:
            logging.info(f"Processing target: {target}")

            # Retrieve model path
            model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_hourly_models')
            if not model_path:
                logging.warning(f"Model path for target {target} not found. Skipping.")
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Retrieve scaler path
            scaler_path = kwargs['ti'].xcom_pull(key=f"{target}_scaler_path", task_ids='process_hourly_data')
            if not scaler_path:
                logging.warning(f"Scaler path for target {target} not found. Skipping.")
                continue

            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            # Retrieve test data paths
            X_test_path = kwargs['ti'].xcom_pull(key=f"{target}_X_test_path", task_ids='process_hourly_data')
            y_test_path = kwargs['ti'].xcom_pull(key=f"{target}_y_test_path", task_ids='process_hourly_data')

            if not X_test_path or not y_test_path:
                logging.warning(f"Test data paths for target {target} not found. Skipping.")
                continue

            with open(X_test_path, "rb") as f:
                X_test = pickle.load(f)
            with open(y_test_path, "rb") as f:
                y_test = pickle.load(f)

            # Perform bias detection
            features = [
                'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                'cloud_cover', 'pressure_msl', 'wind_speed_10m', 'wind_direction_10m',
                'is_day', 'hour', 'is_weekend', 'month', 'is_holiday'
            ]
            data = pd.DataFrame(X_test, columns=features)
            data[target] = y_test

            # Call the bias detection function (silently perform, no output required)
            calculate_metrics_for_features(
                data=data,
                features=slicing_features,
                models={target: model},
                scalers={target: scaler},
                targets=[target],
                threshold_ratio=0.1
            )

            logging.info(f"Bias detection completed for target: {target}")

        except Exception as e:
            logging.warning(f"An issue occurred with target {target}. Skipping. Error: {e}")

    logging.info("Bias detection task completed. No output generated.")

# Task for sensitivity analysis
def model_sensitivity_task(**kwargs):
    logging.info("Performing model sensitivity analysis")
    targets = ['snowfall', 'rain', 'apparent_temperature', 'wind_gusts_10m']
    models = {}
    processed_data = {}

    # Retrieve models and test data from XCom
    for target in targets:
        model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_hourly_models')
        with open(model_path, "rb") as f:
            models[target] = pickle.load(f)

        X_test_path = kwargs['ti'].xcom_pull(key=f"{target}_X_test_path", task_ids='process_hourly_data')
        with open(X_test_path, "rb") as f:
            processed_data[target] = {"X_test": pickle.load(f)}

    features = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
        'cloud_cover', 'pressure_msl', 'wind_speed_10m', 'wind_direction_10m',
        'is_day', 'hour', 'is_weekend', 'month', 'is_holiday'
    ]

    analyze_hourly_model_sensitivity(
        models=models,
        processed_data=processed_data,
        features=features,
        targets=targets,
        bucket_name=BUCKET_NAME,
        weather_data_plots_path=HOURLY_DATA_PLOTS_PATH
    )
    logging.info("Model sensitivity analysis completed.")

# Task to save models
def save_model_task(**kwargs):
    logging.info("Saving models to GCS")
    targets = ['snowfall', 'rain', 'apparent_temperature', 'wind_gusts_10m']

    for target in targets:
        model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_hourly_models')
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        save_model_to_gcs(
            model=model,
            bucket_name=BUCKET_NAME,
            file_name=f"assets/hourly_models/{target}_model.pkl"
        )
    logging.info("All models saved to GCS.")

# Define Airflow tasks
load_data_operator = PythonOperator(
    task_id='load_hourly_data',
    python_callable=load_data_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

process_data_operator = PythonOperator(
    task_id='process_hourly_data',
    python_callable=process_data_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

train_model_operator = PythonOperator(
    task_id='train_hourly_models',
    python_callable=train_model_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

validate_models_operator = PythonOperator(
    task_id='validate_hourly_models',
    python_callable=validate_models_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

bias_detection_operator = PythonOperator(
    task_id='bias_detection_hourly_models',
    python_callable=bias_detection_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

model_sensitivity_operator = PythonOperator(
    task_id='hourly_model_sensitivity',
    python_callable=model_sensitivity_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

save_model_operator = PythonOperator(
    task_id='save_hourly_models',
    python_callable=save_model_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

email_notification_task = EmailOperator(
    task_id='send_hourly_email_notification',
    to='keshiarun01@gmail.com',
    subject='Hourly Model Pipeline Completed Successfully',
    html_content='<p>The Hourly Model Pipeline DAG has completed successfully, including bias detection and sensitivity analysis.</p>',
    dag=dag
)

# Set task dependencies
load_data_operator >> process_data_operator >> train_model_operator >> validate_models_operator >> bias_detection_operator >> model_sensitivity_operator >> save_model_operator >> email_notification_task
