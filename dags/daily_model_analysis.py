import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import logging
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def bias_analysis(models, data, date_features, target_features, output_dir):
    """
    Perform bias analysis by comparing predicted vs. actual values.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for target in target_features:
        logging.info(f"Performing bias analysis for {target}...")
        X = data[date_features]
        y_actual = data[target]
        y_pred = models[target].predict(X)

        # Compute bias metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = mean_squared_error(y_actual, y_pred, squared=False)
        r2 = r2_score(y_actual, y_pred)
        bias = (y_pred - y_actual).mean()  # Average prediction error

        # Store metrics
        results[target] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'bias': bias}

        # Generate bias plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_actual, y=y_pred, alpha=0.7)
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], color='red', linestyle='--')
        plt.title(f"Bias Analysis for {target}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid()
        plot_path = os.path.join(output_dir, f"{target}_bias_plot.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved bias plot for {target} at {plot_path}")

    return results

def sensitivity_analysis(models, data, date_features, target_features, output_dir):
    """
    Perform sensitivity analysis by perturbing date features and observing predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    perturbations = {
        'month': [1, 6, 12],
        'day_of_year': [1, 182, 365],
        'week_of_year': [1, 26, 52]
    }
    results = {}

    for target in target_features:
        logging.info(f"Performing sensitivity analysis for {target}...")
        sensitivities = []
        for feature, values in perturbations.items():
            for value in values:
                perturbed_data = data.copy()
                perturbed_data[feature] = value
                X = perturbed_data[date_features]
                y_pred = models[target].predict(X)
                sensitivities.append({
                    'feature': feature,
                    'value': value,
                    'average_prediction': y_pred.mean()
                })

        # Save sensitivity results
        results[target] = pd.DataFrame(sensitivities)
        sensitivity_path = os.path.join(output_dir, f"{target}_sensitivity.csv")
        results[target].to_csv(sensitivity_path, index=False)
        logging.info(f"Saved sensitivity analysis for {target} at {sensitivity_path}")

    return results

def upload_to_gcs(bucket_name, source_path, destination_blob):
    """
    Upload files to Google Cloud Storage.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(source_path)
        logging.info(f"Uploaded {source_path} to {bucket_name}/{destination_blob}")
    except Exception as e:
        logging.error(f"Error uploading {source_path} to GCS: {e}")
        raise
