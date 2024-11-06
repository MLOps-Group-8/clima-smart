import pandas as pd
from google.cloud import storage
import io
import logging
import pickle

# Constants for Google Cloud Storage and file paths
BUCKET_NAME = 'us-east1-climasmart-fefe9cc2-bucket'
DAILY_DATA_PATH = 'weather_data/engineered_daily_data.csv'
HOURLY_DATA_PATH = 'weather_data/engineered_hourly_data.csv'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_gcs(bucket_name, file_path):
    """Load CSV data directly from GCS into a DataFrame."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_string()
    return pd.read_csv(io.BytesIO(data))

def save_object_to_gcs(bucket_name, object_data, destination_path):
    """Save a Python object to GCS as a pickle file."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    try:
        with io.BytesIO() as f:
            pickle.dump(object_data, f)
            f.seek(0)
            blob.upload_from_file(f, content_type='application/octet-stream')
        logging.info(f"Object saved to GCS at {destination_path}")
    except Exception as e:
        logging.error(f"Failed to save object to GCS at {destination_path}: {e}")

def load_object_from_gcs(bucket_name, source_path):
    """Load a Python object from a pickle file in GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_path)
    try:
        with io.BytesIO(blob.download_as_string()) as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load object from GCS at {source_path}: {e}")
        return None

# Custom test functions

def test_no_nulls(data, dataset_name):
    """Check for null values in the dataset."""
    if data.isnull().sum().sum() > 0:
        logging.error(f"{dataset_name} contains null values.")
    else:
        logging.info(f"No null values found in {dataset_name}.")

def test_positive_temperatures(data, dataset_name):
    """Ensure temperature columns have realistic values."""
    if 'temperature_2m_min' in data.columns:
        if not (data['temperature_2m_min'] >= -100).all():
            logging.error(f"{dataset_name} has unrealistic minimum temperatures.")
        else:
            logging.info(f"All minimum temperatures in {dataset_name} are realistic.")

    if 'temperature_2m_max' in data.columns:
        if not (data['temperature_2m_max'] <= 150).all():
            logging.error(f"{dataset_name} has unrealistic maximum temperatures.")
        else:
            logging.info(f"All maximum temperatures in {dataset_name} are realistic.")

def test_precipitation_non_negative(data, dataset_name):
    """Ensure precipitation values are non-negative."""
    if 'precipitation_sum' in data.columns:
        if not (data['precipitation_sum'] >= 0).all():
            logging.error(f"{dataset_name} has negative precipitation values.")
        else:
            logging.info(f"All precipitation values in {dataset_name} are non-negative.")

def test_schema_similarity(schema1, schema2):
    """Check if two schemas are similar by comparing feature keys and types."""
    schema1_features = schema1.get('features', {})
    schema2_features = schema2.get('features', {})

    if schema1_features != schema2_features:
        logging.warning("Schemas for daily and hourly data do not match.")
    else:
        logging.info("Schemas for daily and hourly data match.")

def validate_weather_data():
    """Run custom data validation checks on daily and hourly data."""
    # Load the engineered data from GCS
    logging.info("Loading engineered daily and hourly data from GCS.")
    daily_data = load_data_from_gcs(BUCKET_NAME, DAILY_DATA_PATH)
    hourly_data = load_data_from_gcs(BUCKET_NAME, HOURLY_DATA_PATH)

    # Run validation checks on daily data
    logging.info("Running validation checks on daily data.")
    test_no_nulls(daily_data, "Daily Data")
    test_positive_temperatures(daily_data, "Daily Data")
    test_precipitation_non_negative(daily_data, "Daily Data")

    # Run validation checks on hourly data
    logging.info("Running validation checks on hourly data.")
    test_no_nulls(hourly_data, "Hourly Data")
    test_positive_temperatures(hourly_data, "Hourly Data")
    test_precipitation_non_negative(hourly_data, "Hourly Data")

    logging.info("Data validation completed.")

def save_schema_and_stats(daily_schema, hourly_schema, daily_stats, hourly_stats):
    """Save schemas and statistics to GCS for future reference."""
    save_object_to_gcs(BUCKET_NAME, daily_schema, 'weather_data_validation/daily_schema.pkl')
    save_object_to_gcs(BUCKET_NAME, hourly_schema, 'weather_data_validation/hourly_schema.pkl')
    save_object_to_gcs(BUCKET_NAME, daily_stats, 'weather_data_validation/daily_stats.pkl')
    save_object_to_gcs(BUCKET_NAME, hourly_stats, 'weather_data_validation/hourly_stats.pkl')

def test_data_quality_and_schema():
    """Test data quality and schema similarity by loading schemas from GCS and running tests."""
    # Load schemas from GCS
    daily_schema = load_object_from_gcs(BUCKET_NAME, 'weather_data_validation/daily_schema.pkl')
    hourly_schema = load_object_from_gcs(BUCKET_NAME, 'weather_data_validation/hourly_schema.pkl')
    
    # Log types for debugging
    logging.info(f"Type of daily_schema: {type(daily_schema)}")
    logging.info(f"Type of hourly_schema: {type(hourly_schema)}")
    
    # Ensure schemas are dictionaries
    if not isinstance(daily_schema, dict) or not isinstance(hourly_schema, dict):
        logging.error("Schemas are not in the expected dictionary format. Exiting validation.")
        return
    
    # Run schema similarity test
    test_schema_similarity(daily_schema, hourly_schema)
