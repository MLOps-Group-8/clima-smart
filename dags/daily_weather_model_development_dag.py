from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import json 
import os
import pickle
from utils import save_model_to_gcs, read_data_from_gcs
from dailymodeltraining import (
    process_data,
    run_model_training,
)
from dailymodelbiasdetection import run_bias_detection_workflow, plot_bias_metrics
from dailymodelsensitivity import perform_feature_importance_analysis, perform_hyperparameter_sensitivity_analysis
from constants import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
dag3 = DAG(
    'daily_weather_model_development_pipeline_v2',
    default_args=default_args,
    description='DAG for running the model development pipeline on daily weather data',
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=False
)

# Temporary directory for saving data and models
TEMP_DIR = "/tmp/airflow_daily_model_pipeline"
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
    data = read_data_from_gcs(BUCKET_NAME, ENGINEERED_DAILY_DATA_PATH)
    data_path = os.path.join(TEMP_DIR, "engineered_data.pkl")

    logging.info(f"Saving loaded data to {data_path}")
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file could not be created at {data_path}")

    kwargs['ti'].xcom_push(key='engineered_data_path', value=data_path)
    logging.info("Data loading task completed.")

# Task to process data
def process_data_task(**kwargs):
    logging.info("Processing data")

    raw_data_path = kwargs['ti'].xcom_pull(key='engineered_data_path', task_ids='load_data')
    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)

    features = [
        'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 'rain_sum',
        'showers_sum', 'daylight_duration', 'precipitation_sum', 'temperature_range',
        'diurnal_temp_range', 'precipitation_intensity'
    ]
    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum']

    # Remove scalers from process_data function
    processed_data, scaler_features, _ = process_data(raw_data, features, targets)

    for target, splits in processed_data.items():
        for split, data_split in splits.items():
            split_path = os.path.join(TEMP_DIR, f"{target}_{split}.pkl")
            with open(split_path, "wb") as f:
                pickle.dump(data_split, f)

            # Log feature information
            if isinstance(data_split, np.ndarray):
                logging.info(f"Saving processed data for {target} - {split}: {data_split.shape}")
                logging.info(f"Data split is a NumPy array; columns cannot be displayed.")
            elif isinstance(data_split, pd.DataFrame):
                logging.info(f"Saving processed data for {target} - {split}: {data_split.shape}")
                logging.info(f"Features in {target} - {split}: {list(data_split.columns)}")
            else:
                logging.warning(f"Unexpected data type for {target} - {split}: {type(data_split)}")

            kwargs['ti'].xcom_push(key=f"{target}_{split}_path", value=split_path)

    # Remove handling of target scalers entirely
    logging.info("Data processing task completed.")
    
def train_model_task(**kwargs):
    logging.info("Starting model training with hyperparameter optimization")

    # Retrieve the data path from XCom
    data_path = kwargs['ti'].xcom_pull(key='engineered_data_path', task_ids='load_data')

    if not data_path:
        raise ValueError("No data path found in XCom for 'engineered_data_path'. Ensure 'load_data_task' pushed the correct path.")

    logging.info(f"Data path retrieved from XCom: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at path: {data_path}")

    # Define the output path for the combined models file
    combined_output_file = os.path.join(TEMP_DIR, "all_models.pkl")

    # Pass the data to the model_training function
    models_file_path = run_model_training(data_path, combined_output_file, task_instance=kwargs['ti'])

    # Push the path of the combined models file to XCom
    kwargs['ti'].xcom_push(key='models_file_path', value=models_file_path)
    logging.info(f"Model training task completed successfully. Models saved at {models_file_path}")


# Task for bias detection (remove dependency on scalers)
def bias_detection_task(**kwargs):
    logging.info("Starting bias detection")

    # Load processed data path
    data_path = kwargs['ti'].xcom_pull(key='engineered_data_path', task_ids='load_data')
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # Load models
    models = {}
    for target in ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum']:
        model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_model')
        with open(model_path, "rb") as f:
            models[target] = pickle.load(f)
    
    features = [
    'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 'rain_sum',
    'showers_sum', 'daylight_duration', 'precipitation_sum', 'temperature_range',
    'diurnal_temp_range', 'precipitation_intensity'
    ]
    
    # Define slicing features and their configurations
    slicing_definitions = {
        'temperature_2m_max_bin': {'column': 'temperature_2m_max', 'bins': 4, 'labels': ['cold', 'cool', 'warm', 'hot']},
        'precipitation_sum_bin': {'column': 'precipitation_sum', 'bins': 4, 'labels': ['low', 'medium-low', 'medium-high', 'high']}
        }

    # List of target variables
    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum']

    # Dictionary to store metrics for all targets
    all_metrics = {}

    # Loop through each target
    for target in targets:
        logging.info(f"Starting bias detection workflow for target: {target}")

        # Run bias detection workflow for the current target
        metrics = run_bias_detection_workflow(data, features, target, models[target], slicing_definitions)

        # Store the metrics for the current target
        all_metrics[target] = metrics

    # Print and visualize bias detection metrics for all targets
    for target, metrics in all_metrics.items():
        print(f"\nBias Metrics for {target}:")
        for slicing_feature, (overall, by_group) in metrics.items():
            print(f"  Slicing Feature: {slicing_feature}")
            print(f"    Overall RMSE: {overall}")
            print(f"    By Group RMSE: {by_group}")
            plot_bias_metrics(metrics, slicing_features=[slicing_feature])


    logging.info("Bias detection task completed successfully.")

    
def model_sensitivity_task(bucket_name=BUCKET_NAME, **kwargs):
    logging.info("Performing model sensitivity analysis")

    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum']
    features = [
        'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 'rain_sum',
        'showers_sum', 'daylight_duration', 'precipitation_sum', 'temperature_range',
        'diurnal_temp_range', 'precipitation_intensity'
    ]
    models = {}
    processed_data = {}

    for target in targets:
        # Load model
        model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_model')
        if not model_path or not os.path.exists(model_path):
            logging.error(f"Model path for {target} is missing or invalid: {model_path}")
            continue
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            models[target] = model_data.get("model")
            if models[target] is None:
                logging.error(f"Failed to extract model for {target}")
                continue

        # Load test data
        X_test_path = kwargs['ti'].xcom_pull(key=f"{target}_X_test_path", task_ids='process_data')
        if not X_test_path or not os.path.exists(X_test_path):
            logging.error(f"Test data for {target} is missing or invalid: {X_test_path}")
            continue
        with open(X_test_path, "rb") as f:
            X_test = pickle.load(f)[features]  # Filter only relevant features

        # Perform feature importance analysis
        try:
            perform_feature_importance_analysis(
                model=models[target],
                X_test=X_test,
                feature_names=features,
                target=target,
                bucket_name=bucket_name,
            )
        except Exception as e:
            logging.error(f"Feature importance analysis failed for {target}: {e}")

        # Load trials and perform hyperparameter sensitivity analysis
        trial_path = kwargs['ti'].xcom_pull(key=f"{target}_trials_path", task_ids='train_model')
        if not trial_path or not os.path.exists(trial_path):
            logging.error(f"Trials data for {target} is missing or invalid: {trial_path}")
            continue
        with open(trial_path, "rb") as f:
            trials = pickle.load(f)

        try:
            perform_hyperparameter_sensitivity_analysis(trials, target, bucket_name)
        except Exception as e:
            logging.error(f"Hyperparameter sensitivity analysis failed for {target}: {e}")

    logging.info("Model sensitivity analysis task completed successfully.")
    
# Task to save best models to GCS
def save_models_to_gcs_task(**kwargs):
    logging.info("Saving best models to GCS")

    # Retrieve the best model paths from XCom
    best_model_paths = kwargs['ti'].xcom_pull(key='models_file_path', task_ids='train_model')

    # Define the GCS path for the model
    gcs_path = f"assets/daily_models/best_model.pkl"

    # Save the model to GCS
    save_model_to_gcs(
    model=best_model_paths,
    bucket_name=BUCKET_NAME
    )
    logging.info(f"Best mode saved to GCS at {gcs_path}")

# Define Airflow tasks
load_data_operator = PythonOperator(
    task_id='load_data',
    python_callable=load_data_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag3
)

process_data_operator = PythonOperator(
    task_id='process_data',
    python_callable=process_data_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag3
)

train_model_operator = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    execution_timeout=timedelta(minutes=10),
    dag=dag3
)

bias_detection_operator = PythonOperator(
    task_id='bias_detection',
    python_callable=bias_detection_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag3
)

# Define the task in the DAG
model_sensitivity_operator = PythonOperator(
    task_id='model_sensitivity',
    python_callable=model_sensitivity_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag3
)

save_best_model_operator = PythonOperator(
    task_id='save_best_model',
    python_callable=save_models_to_gcs_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag3
)

email_notification_task = EmailOperator(
    task_id='send_email_notification',
    to='darshan.webjaguar@gmail.com',
    subject='Daily Model Development Pipeline Completed Successfully',
    html_content='<p>The Daily Model Development Pipeline DAG has completed successfully.</p>',
    dag=dag3
)

# Set task dependencies
load_data_operator >> process_data_operator >> train_model_operator >> bias_detection_operator >> model_sensitivity_operator >> save_best_model_operator >> email_notification_task