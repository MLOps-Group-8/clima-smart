import pandas as pd
import io
import logging
from google.cloud import storage
import pickle
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_data_from_gcs(bucket_name, blob_name):
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

def save_data_to_gcs(df, bucket_name, file_name):
    """Save DataFrame as a CSV file in Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    blob.upload_from_file(output, content_type='text/csv')
    logging.info(f"Saved {file_name} to GCS.")

def save_plot_to_gcs(bucket_name, plot_name):
    """Save the current matplotlib plot to Google Cloud Storage in a specific folder."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"weather_data_plots/{plot_name}.png")
    plot_image = io.BytesIO()
    plt.savefig(plot_image, format='png')
    plot_image.seek(0)
    blob.upload_from_file(plot_image, content_type='image/png')
    logging.info(f"Plot {plot_name} saved to GCS in folder weather_data_plots.")

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