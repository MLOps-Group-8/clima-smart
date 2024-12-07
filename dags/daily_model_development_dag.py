from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
import os
import logging
from daily_model_training_v2 import train_and_save_models, load_models, predict_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from constants import *

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
    dag_id='daily_model_development_pipeline',
    default_args=default_args,
    description='Automates training, prediction, monitoring, and retraining for weather forecasting',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['weather', 'forecasting', 'daily'],
)

# Configuration
LOCAL_TRAIN_FILE = '/tmp/daily_weather_data.csv'
MODEL_DIR = '/tmp/models'
TARGET_FEATURES = ['apparent_temperature_max', 'humidity', 'wind_speed']
METRIC_THRESHOLDS = {'rmse': 5.0, 'r2': 0.8}  # Example thresholds for retraining

def update_train_file():
    """Download the daily training file from GCS."""
    try:
        logging.info(f"Downloading daily training file from {ENGINEERED_DAILY_DATA_PATH}...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(ENGINEERED_DAILY_DATA_PATH)
        blob.download_to_filename(LOCAL_TRAIN_FILE)
        logging.info(f"Daily training file saved to {LOCAL_TRAIN_FILE}")
    except Exception as e:
        logging.error(f"Failed to download daily training file: {e}")
        raise

def train_models():
    """Train models for all target features and save them."""
    try:
        logging.info("Starting model training...")
        train_and_save_models(
            data_path=LOCAL_TRAIN_FILE,
            model_dir=MODEL_DIR,
            target_features=TARGET_FEATURES
        )
        logging.info("Model training completed.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def upload_models_to_gcs():
    """Upload trained models to GCS."""
    try:
        logging.info("Uploading models to GCS...")
        storage_client = storage.Client()
        for target in TARGET_FEATURES:
            local_model_path = os.path.join(MODEL_DIR, f"{target}_model.json")
            remote_model_path = f"models/daily/{target}_model.json"
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(remote_model_path)
            blob.upload_from_filename(local_model_path)
            logging.info(f"Uploaded {local_model_path} to {remote_model_path}")
    except Exception as e:
        logging.error(f"Error uploading models to GCS: {e}")
        raise

def predict_for_today():
    """Predict weather features for today using trained models."""
    try:
        logging.info("Starting predictions...")
        models = load_models(MODEL_DIR, TARGET_FEATURES)
        today = datetime.now().strftime('%Y-%m-%d')
        predictions = predict_features(models, today, TARGET_FEATURES)
        logging.info(f"Predictions for {today}: {predictions}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

def monitor_model_performance():
    """Monitor model performance by comparing predictions to actual values and trigger retraining if needed."""
    try:
        logging.info("Starting model monitoring...")
        models = load_models(MODEL_DIR, TARGET_FEATURES)

        # Load the updated training data
        logging.info(f"Loading actual data from {LOCAL_TRAIN_FILE}...")
        data = pd.read_csv(LOCAL_TRAIN_FILE)
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        data['day_of_year'] = data['date'].dt.day_of_year
        data['week_of_year'] = data['date'].dt.isocalendar().week
        data['is_weekend'] = data['date'].dt.weekday >= 5
        date_features = ['month', 'day_of_year', 'week_of_year', 'is_weekend']

        # Evaluate each model
        retrain_needed = False
        performance_metrics = {}
        for target in TARGET_FEATURES:
            logging.info(f"Evaluating model for {target}...")
            X = data[date_features]
            y_actual = data[target]
            y_pred = models[target].predict(X)

            # Calculate metrics
            rmse = mean_squared_error(y_actual, y_pred, squared=False)
            mae = mean_absolute_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)
            performance_metrics[target] = {'rmse': rmse, 'mae': mae, 'r2': r2}

            # Log metrics
            logging.info(f"{target} - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

            # Check thresholds for retraining
            if rmse > METRIC_THRESHOLDS['rmse'] or r2 < METRIC_THRESHOLDS['r2']:
                logging.warning(f"{target}: Metrics exceeded thresholds. RMSE: {rmse}, R²: {r2}")
                retrain_needed = True

        # Trigger retraining if needed
        if retrain_needed:
            logging.info("Performance degraded. Triggering retraining...")
            train_models()
            upload_models_to_gcs()
            logging.info("Retraining completed.")
        else:
            logging.info("Model performance is within acceptable limits.")

    except Exception as e:
        logging.error(f"Error during model monitoring: {e}")
        raise

# Define tasks
update_train_file_task = PythonOperator(
    task_id='update_train_file',
    python_callable=update_train_file,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

upload_models_task = PythonOperator(
    task_id='upload_models_to_gcs',
    python_callable=upload_models_to_gcs,
    dag=dag,
)

predict_for_today_task = PythonOperator(
    task_id='predict_for_today',
    python_callable=predict_for_today,
    dag=dag,
)

monitor_model_performance_task = PythonOperator(
    task_id='monitor_model_performance',
    python_callable=monitor_model_performance,
    dag=dag,
)

# Define task dependencies
update_train_file_task >> train_models_task >> upload_models_task >> predict_for_today_task >> monitor_model_performance_task
