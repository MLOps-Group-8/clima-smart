import pandas as pd
import io
import logging
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv_from_gcs(bucket_name, blob_name):
    """
    Reads a CSV file directly into a pandas DataFrame from Google Cloud Storage.

    Args:
    bucket_name (str): Name of the GCS bucket.
    blob_name (str): The specific blob (file path in GCS) to read.

    Returns:
    pandas.DataFrame: Data read from the CSV file.
    """
    logging.info(f"Attempting to read from GCS: {bucket_name}/{blob_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()
    df = pd.read_csv(io.StringIO(data))
    logging.info(f"Successfully read data from {blob_name}")
    return df
