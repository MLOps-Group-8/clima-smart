from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import logging
import os
import pickle
from hourlymodeldag import (
    load_data_from_gcs,
    process_data,
    build_model,
    train_model,
    evaluate_model,
    save_model
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GCS bucket and file path
BUCKET_NAME = 'clima-smart-data-collection'
ENGINEERED_HOURLY_DATA_PATH = 'weather_data/engineered_hourly_data.csv'

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
    'ModelDevelopmentPipeline',
    default_args=default_args,
    description='DAG for running the model development pipeline',
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=False
)

# Temporary directory for saving data and models
TEMP_DIR = "/tmp/airflow_model_pipeline"
MODEL_DIR = os.path.abspath(os.path.join(os.getcwd(), "/Users/keshikaarunkumar/mlops/clima-smart/clima-smart/assets/models"))
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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

    # Pull raw data from XCom
    raw_data_path = kwargs['ti'].xcom_pull(key='raw_data_path', task_ids='load_data')
    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)

    # Call the actual process_data function from model_development_pipeline_functions
    from hourlymodeldag import process_data
    X_train, X_test, y_train, y_test = process_data(raw_data)

    # Save processed data
    X_train_path = os.path.join(TEMP_DIR, "X_train.pkl")
    X_test_path = os.path.join(TEMP_DIR, "X_test.pkl")
    y_train_path = os.path.join(TEMP_DIR, "y_train.pkl")
    y_test_path = os.path.join(TEMP_DIR, "y_test.pkl")

    with open(X_train_path, "wb") as f:
        pickle.dump(X_train, f)
    with open(X_test_path, "wb") as f:
        pickle.dump(X_test, f)
    with open(y_train_path, "wb") as f:
        pickle.dump(y_train, f)
    with open(y_test_path, "wb") as f:
        pickle.dump(y_test, f)

    # Push paths to XCom
    kwargs['ti'].xcom_push(key='X_train_path', value=X_train_path)
    kwargs['ti'].xcom_push(key='X_test_path', value=X_test_path)
    kwargs['ti'].xcom_push(key='y_train_path', value=y_train_path)
    kwargs['ti'].xcom_push(key='y_test_path', value=y_test_path)

    logging.info("Data processing task completed.")

# Task: Build Model
def build_model_task(**kwargs):
    logging.info("Building model")
    X_train_path = kwargs['ti'].xcom_pull(key='X_train_path', task_ids='process_data')
    with open(X_train_path, "rb") as f:
        X_train = pickle.load(f)

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model_path = os.path.join(TEMP_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    kwargs['ti'].xcom_push(key='model_path', value=model_path)
    logging.info("Model building task completed.")

# Task: Train Model
def train_model_task(**kwargs):
    logging.info("Training model")
    X_train_path = kwargs['ti'].xcom_pull(key='X_train_path', task_ids='process_data')
    X_test_path = kwargs['ti'].xcom_pull(key='X_test_path', task_ids='process_data')
    y_train_path = kwargs['ti'].xcom_pull(key='y_train_path', task_ids='process_data')
    y_test_path = kwargs['ti'].xcom_pull(key='y_test_path', task_ids='process_data')
    model_path = kwargs['ti'].xcom_pull(key='model_path', task_ids='build_model')

    with open(X_train_path, "rb") as f:
        X_train = pickle.load(f)
    with open(X_test_path, "rb") as f:
        X_test = pickle.load(f)
    with open(y_train_path, "rb") as f:
        y_train = pickle.load(f)
    with open(y_test_path, "rb") as f:
        y_test = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    trained_model = train_model(X_train, X_test, y_train, y_test)
    trained_model_path = os.path.join(TEMP_DIR, "trained_model.pkl")
    with open(trained_model_path, "wb") as f:
        pickle.dump(trained_model, f)

    kwargs['ti'].xcom_push(key='trained_model_path', value=trained_model_path)
    logging.info("Model training task completed.")

# Task: Evaluate Model
def evaluate_model_task(**kwargs):
    logging.info("Evaluating model")
    X_test_path = kwargs['ti'].xcom_pull(key='X_test_path', task_ids='process_data')
    y_test_path = kwargs['ti'].xcom_pull(key='y_test_path', task_ids='process_data')
    trained_model_path = kwargs['ti'].xcom_pull(key='trained_model_path', task_ids='train_model')

    with open(X_test_path, "rb") as f:
        X_test = pickle.load(f)
    with open(y_test_path, "rb") as f:
        y_test = pickle.load(f)
    with open(trained_model_path, "rb") as f:
        trained_model = pickle.load(f)

    rmse, r2 = evaluate_model(trained_model, X_test, y_test)
    logging.info(f"Model evaluation completed. RMSE: {rmse}, R^2: {r2}")

# Task: Save Model
# Task: Save Model
def save_model_task(**kwargs):
    """
    Save the trained model to the assets/models/ directory.
    """
    logging.info("Saving model")
    
    # Get the trained model path from XCom
    trained_model_path = kwargs['ti'].xcom_pull(key='trained_model_path', task_ids='train_model')
    
    # Load the trained model from the temporary directory
    with open(trained_model_path, "rb") as f:
        trained_model = pickle.load(f)
    
    # Save the model to assets/models/ directory
    model_path = os.path.join(MODEL_DIR, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(trained_model, f)
    
    logging.info(f"Model saved successfully at {model_path}.")
    kwargs['ti'].xcom_push(key='saved_model_path', value=model_path)  # Optional, in case downstream tasks need the path

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

build_model_operator = PythonOperator(
    task_id='build_model',
    python_callable=build_model_task,
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

evaluate_model_operator = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
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
    to='keshiarun01@gmail.com',
    subject='Model Development Pipeline Completed Successfully',
    html_content='<p>The Model Development Pipeline DAG has completed successfully.</p>',
    dag=dag
)

# Set task dependencies
load_data_operator >> process_data_operator >> build_model_operator >> train_model_operator >> evaluate_model_operator >> save_model_operator >> email_notification_task