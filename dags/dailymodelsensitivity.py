import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import io
import logging
from google.cloud import storage


def upload_to_gcs(bucket_name, destination_path, plt):
    """
    Upload a Matplotlib plot to Google Cloud Storage as an image.

    Args:
        bucket_name (str): Name of the GCS bucket.
        destination_path (str): Path in the GCS bucket to save the plot.
        plt: Matplotlib plot object.

    Returns:
        None
    """
    logging.info(f"Uploading plot to GCS bucket {bucket_name} at {destination_path}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_path)

    # Save the plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    # Upload the bytes buffer to GCS
    blob.upload_from_file(buffer, content_type="image/png")
    logging.info(f"Plot uploaded successfully to gs://{bucket_name}/{destination_path}")


def calculate_shap_values(model, X_test, feature_names):
    """
    Calculate SHAP values for the given model and test data.

    Args:
        model: Trained XGBoost model.
        X_test (numpy.ndarray): Test data used for SHAP analysis.
        feature_names (list): Names of the features.

    Returns:
        shap_values: SHAP values for the test data.
    """
    logging.info("Calculating SHAP values.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values


def generate_shap_plots(bucket_name, shap_values, X_test, feature_names, target, weather_data_plots_path):
    """
    Generate SHAP summary plots and save them to GCS.

    Args:
        bucket_name (str): Name of the GCS bucket.
        shap_values: SHAP values for the test data.
        X_test (numpy.ndarray): Test data used for SHAP analysis.
        feature_names (list): Names of the features.
        target (str): Target variable being analyzed.
        weather_data_plots_path (str): GCS path to save the SHAP plots.

    Returns:
        None
    """
    logging.info(f"Generating SHAP plots for target: {target}")

    # Summary Plot - Bar
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    bar_plot_path = f"{weather_data_plots_path}SHAP_Bar_Summary_{target}.png"
    upload_to_gcs(bucket_name, bar_plot_path, plt)
    plt.close()

    # Summary Plot - Detailed
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    detailed_plot_path = f"{weather_data_plots_path}SHAP_Detailed_Summary_{target}.png"
    upload_to_gcs(bucket_name, detailed_plot_path, plt)
    plt.close()

    logging.info(f"SHAP plots for target {target} uploaded to GCS.")


def analyze_model_sensitivity(models, processed_data, features, targets, bucket_name, weather_data_plots_path):
    """
    Perform model sensitivity analysis using SHAP and upload plots to GCS.

    Args:
        models (dict): Dictionary of trained models, keyed by target name.
        processed_data (dict): Dictionary containing test data for each target.
        features (list): Feature names used in the model.
        targets (list): List of target variables being analyzed.
        bucket_name (str): Name of the GCS bucket.
        weather_data_plots_path (str): GCS path to save the SHAP plots.

    Returns:
        None
    """
    logging.info("Starting model sensitivity analysis.")

    for target in targets:
        logging.info(f"Analyzing sensitivity for target: {target}")
        model = models[target]
        X_test = processed_data[target]["X_test"]

        # Calculate SHAP values
        shap_values = calculate_shap_values(model, X_test, features)

        # Generate SHAP plots and upload them to GCS
        generate_shap_plots(bucket_name, shap_values, X_test, features, target, weather_data_plots_path)

    logging.info("Model sensitivity analysis completed.")
