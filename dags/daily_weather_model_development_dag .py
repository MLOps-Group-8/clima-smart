from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import os
import pickle
from utils import save_model_to_gcs, read_data_from_gcs
from daily_model_training import (
    process_data,
    train_model,
)
from daily_model_validation import evaluate_and_visualize_model
from daily_model_bias_detection import calculate_metrics_for_features, bin_column
from daily_model_sensitivity import analyze_model_sensitivity
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
dag = DAG(
    'daily_weather_model_development_pipeline',
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
    data_path = os.path.join(TEMP_DIR, "raw_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    kwargs['ti'].xcom_push(key='raw_data_path', value=data_path)
    logging.info("Data loading task completed.")

# Task to process data
def process_data_task(**kwargs):
    logging.info("Processing data")

    raw_data_path = kwargs['ti'].xcom_pull(key='raw_data_path', task_ids='load_data')
    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)

    features = [
        'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 'rain_sum',
        'showers_sum', 'daylight_duration', 'precipitation_sum', 'temperature_range',
        'diurnal_temp_range', 'precipitation_intensity'
    ]
    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum', 
               'precipitation_sum', 'precipitation_hours']

    processed_data, scaler_features, target_scalers = process_data(raw_data, features, targets)

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

    for target, scaler in target_scalers.items():
        scaler_path = os.path.join(TEMP_DIR, f"{target}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        kwargs['ti'].xcom_push(key=f"{target}_scaler_path", value=scaler_path)

    logging.info("Data processing task completed.")

    
# Task to train model
def train_model_task(**kwargs):
    logging.info("Training models for all targets")
    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum', 
               'precipitation_sum', 'precipitation_hours']

    for target in targets:
        # Pull data paths from XCom
        X_train_path = kwargs['ti'].xcom_pull(key=f"{target}_X_train_path", task_ids='process_data')
        X_val_path = kwargs['ti'].xcom_pull(key=f"{target}_X_val_path", task_ids='process_data')
        y_train_path = kwargs['ti'].xcom_pull(key=f"{target}_y_train_path", task_ids='process_data')
        y_val_path = kwargs['ti'].xcom_pull(key=f"{target}_y_val_path", task_ids='process_data')

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

# Task to validate models and generate plots
def validate_models_task(**kwargs):
    logging.info("Validating models and generating visualizations")

    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum', 
               'precipitation_sum', 'precipitation_hours']
    
    # Retrieve processed data and scalers from XCom for each target
    processed_data = {}
    scalers = {}

    for target in targets:
        processed_data[target] = {}
        
        # Retrieve processed data splits for the target from XCom
        for split in ['X_val', 'X_test', 'y_val', 'y_test']:
            split_path = kwargs['ti'].xcom_pull(key=f"{target}_{split}_path", task_ids='process_data')
            with open(split_path, "rb") as f:
                processed_data[target][split] = pickle.load(f)
        
        # Retrieve scaler for the target
        scaler_path = kwargs['ti'].xcom_pull(key=f"{target}_scaler_path", task_ids='process_data')
        with open(scaler_path, "rb") as f:
            scalers[target] = pickle.load(f)

        # Retrieve the model for the target
        model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_model')
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Evaluate and visualize the model for the current target
        evaluate_and_visualize_model(
            model=model,
            processed_data=processed_data,  # Pass all processed data; function uses only required parts
            targets=[target],               # Single target to match function expectations
            scalers=scalers,                # Pass all scalers; function uses only required parts
            bucket_name=BUCKET_NAME,
            WEATHER_DATA_PLOTS_PATH=WEATHER_DATA_PLOTS_PATH
        )
        logging.info(f"Model validation and visualization for target {target} completed.")
    
    logging.info("All models validated and visualized successfully.")

# Task to detect bias
def bias_detection_task(**kwargs):
    logging.info("Performing bias detection")

    slicing_features = ['temperature_2m_max', 'precipitation_sum']  # Features to slice by
    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum',
               'precipitation_sum', 'precipitation_hours']
    models = {}
    scalers = {}

    # Load models and scalers
    for target in targets:
        model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_model')
        with open(model_path, "rb") as f:
            models[target] = pickle.load(f)

        scaler_path = kwargs['ti'].xcom_pull(key=f"{target}_scaler_path", task_ids='process_data')
        with open(scaler_path, "rb") as f:
            scalers[target] = pickle.load(f)

    # Prepare processed data
    for target in targets:
        X_test_path = kwargs['ti'].xcom_pull(key=f"{target}_X_test_path", task_ids='process_data')
        y_test_path = kwargs['ti'].xcom_pull(key=f"{target}_y_test_path", task_ids='process_data')

        with open(X_test_path, "rb") as f:
            X_test = pickle.load(f)
        with open(y_test_path, "rb") as f:
            y_test = pickle.load(f)

        # Recreate DataFrame with all features and target
        features = [
            'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 'rain_sum',
            'showers_sum', 'daylight_duration', 'precipitation_sum', 'temperature_range',
            'diurnal_temp_range', 'precipitation_intensity'
        ]
        data = pd.DataFrame(X_test, columns=features)
        data[target] = y_test  # Add target column to the DataFrame

        # Ensure slicing features exist in the dataset
        missing_features = [feature for feature in slicing_features if feature not in data.columns]
        if missing_features:
            logging.error(f"Slicing features {missing_features} are missing in the processed data for {target}")
            continue

        logging.info(f"Data columns for target {target}: {list(data.columns)}")

        # Perform bias detection
        try:
            metrics_and_interpretations = calculate_metrics_for_features(
                data=data,
                features=slicing_features,
                models={target: models[target]},
                scalers={target: scalers[target]},
                targets=[target],  # Pass the current target
                threshold_ratio=0.1
            )

            # Log interpretations
            for interpretation in metrics_and_interpretations["interpretations"]:
                logging.info(interpretation)

        except ValueError as e:
            logging.error(f"Bias detection failed for target {target}: {str(e)}")
            logging.warning(f"Skipping bias detection for {target} and continuing the pipeline.")

    logging.info("Bias detection task completed.")
    
# Task to perform model sensitivity analysis
def model_sensitivity_task(**kwargs):
    logging.info("Performing model sensitivity analysis using SHAP")

    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum',
               'precipitation_sum', 'precipitation_hours']
    models = {}
    processed_data = {}

    # Load models and processed data
    for target in targets:
        # Load the model
        model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_model')
        with open(model_path, "rb") as f:
            models[target] = pickle.load(f)

        # Load the processed test data
        X_test_path = kwargs['ti'].xcom_pull(key=f"{target}_X_test_path", task_ids='process_data')
        with open(X_test_path, "rb") as f:
            X_test = pickle.load(f)

        # Store test data for SHAP analysis
        processed_data[target] = {"X_test": X_test}

    # Perform sensitivity analysis for all targets
    features = [
        'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 'rain_sum',
        'showers_sum', 'daylight_duration', 'precipitation_sum', 'temperature_range',
        'diurnal_temp_range', 'precipitation_intensity'
    ]
    analyze_model_sensitivity(
        models=models,
        processed_data=processed_data,
        features=features,
        targets=targets,
        bucket_name=BUCKET_NAME,
        weather_data_plots_path=WEATHER_DATA_PLOTS_PATH
    )
    logging.info("Model sensitivity analysis task completed.")

    
# Task to save model
def save_model_task(**kwargs):
    logging.info("Saving models to GCS")
    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum', 
               'precipitation_sum', 'precipitation_hours']

    for target in targets:
        model_path = kwargs['ti'].xcom_pull(key=f"{target}_model_path", task_ids='train_model')
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Save the model to GCS
        save_model_to_gcs(
            model=model,
            bucket_name=BUCKET_NAME,
            file_name=f"assets/daily_models/{target}_model.pkl"
        )
    logging.info("All models saved to GCS.")

# Define Airflow tasks
load_data_operator = PythonOperator(
    task_id='load_data',
    python_callable=load_data_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

process_data_operator = PythonOperator(
    task_id='process_data',
    python_callable=process_data_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

train_model_operator = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

validate_models_operator = PythonOperator(
    task_id='validate_models',
    python_callable=validate_models_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

bias_detection_operator = PythonOperator(
    task_id='bias_detection',
    python_callable=bias_detection_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

# Define the task in the DAG
model_sensitivity_operator = PythonOperator(
    task_id='model_sensitivity',
    python_callable=model_sensitivity_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

save_model_operator = PythonOperator(
    task_id='save_model',
    python_callable=save_model_task,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag
)

email_notification_task = EmailOperator(
    task_id='send_email_notification',
    to='darshan.webjaguar@gmail.com',
    subject='Daily Model Development Pipeline Completed Successfully',
    html_content='<p>The Daily Model Development Pipeline DAG has completed successfully.</p>',
    dag=dag
)

# Set task dependencies
load_data_operator >> process_data_operator >> train_model_operator >> validate_models_operator >> bias_detection_operator >> model_sensitivity_operator >> save_model_operator >> email_notification_task