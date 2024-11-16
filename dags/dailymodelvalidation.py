import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from google.cloud import storage
import io
import os
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_and_visualize_model(model, processed_data, scalers, targets, bucket_name, WEATHER_DATA_PLOTS_PATH = 'weather_data_plots/'):
    """
    Validate the model with visualizations and save plots to a GCS bucket.

    Args:
        model: Trained model.
        processed_data (dict): Processed data split for each target variable.
        scalers (dict): Scalers for each target variable.
        targets (list): List of target variables.
        bucket_name (str): GCS bucket name.
        plot_path (str): Path inside GCS to save the plots.

    Returns:
        None
    """
    logging.info("Starting model validation and visualization")
    
    for target in targets:
        logging.info(f"Validating model for target: {target}")
        
        # Extract the data and scaler
        X_val = processed_data[target]["X_val"]
        y_val = processed_data[target]["y_val"]
        X_test = processed_data[target]["X_test"]
        y_test = processed_data[target]["y_test"]
        scaler_target = scalers[target]

        # Make predictions
        y_val_pred = model.predict(xgb.DMatrix(X_val))
        y_test_pred = model.predict(xgb.DMatrix(X_test))

        # Inverse transform the predictions and actuals
        y_val_actual = scaler_target.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_val_pred_actual = scaler_target.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
        y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_test_pred_actual = scaler_target.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

        # Evaluate the model
        def evaluate_model(y_true, y_pred, label="Set"):
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            logging.info(f"{label} Metrics for {target}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            return {"RMSE": rmse, "MAE": mae, "R2": r2}

        evaluate_model(y_val_actual, y_val_pred_actual, label=f"Validation Set ({target})")
        evaluate_model(y_test_actual, y_test_pred_actual, label=f"Test Set ({target})")

        # Residual plot
        residuals = y_test_actual - y_test_pred_actual
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30)
        plt.title(f"Residual Distribution ({target})")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        residual_plot_path = f"{WEATHER_DATA_PLOTS_PATH}Residual_Distribution_{target}.png"
        save_plot_to_gcs(plt, bucket_name, residual_plot_path)
        plt.close()

        # Actual vs Predicted plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_actual, y_test_pred_actual, alpha=0.6)
        plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
        plt.title(f"Actual vs Predicted: {target}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        actual_vs_predicted_path = f"{WEATHER_DATA_PLOTS_PATH}Actual_vs_Predicted_{target}.png"
        save_plot_to_gcs(plt, bucket_name, actual_vs_predicted_path)
        plt.close()

        logging.info(f"Validation plots for {target} saved to GCS.")

    logging.info("Model validation and visualization completed")

def save_plot_to_gcs(plot, bucket_name, plot_path):
    """
    Save a generated plot to a GCS bucket.

    Args:
        plot (matplotlib.pyplot): The plot object to save.
        bucket_name (str): Name of the GCS bucket.
        plot_path (str): Path inside GCS to save the plot.

    Returns:
        None
    """
    logging.info(f"Saving plot to GCS bucket: {bucket_name}, path: {plot_path}")

    # Initialize the GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(plot_path)

    # Save plot to a BytesIO buffer
    buffer = io.BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)

    # Upload buffer to GCS
    blob.upload_from_file(buffer, content_type='image/png')
    logging.info(f"Plot successfully saved to GCS: gs://{bucket_name}/{plot_path}")
