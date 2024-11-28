import os
import json
import logging
import pickle
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Temporary directory for saving data and models
TEMP_DIR = "/tmp/airflow_daily_model_pipeline"
os.makedirs(TEMP_DIR, exist_ok=True)

MLFLOW_URI = "http://172.18.0.4:5000"

def setup_mlflow():
    """
    Set up MLflow tracking inside tasks.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("daily_weather_model_pipeline")

def load_metrics(file_path):
    """
    Load model metrics from a JSON file.
    Args:
        file_path (str): Path to the metrics file.
    Returns:
        dict: Metrics data.
    """
    try:
        with open(file_path, "r") as f:
            metrics = json.load(f)
        logger.info(f"Loaded metrics from {file_path}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to load metrics from {file_path}: {e}")
        return None

def simplify_metrics(metrics):
    """
    Flatten nested dictionary metrics to make them JSON-serializable.
    Args:
        metrics (dict): Metrics data.
    Returns:
        dict: Simplified metrics.
    """
    simplified_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                simplified_metrics[f"{key}_{sub_key}"] = sub_value
        else:
            simplified_metrics[key] = value
    return simplified_metrics

def log_best_model_to_mlflow(model_path, metrics):
    """
    Log and register the best model and its metrics in MLflow.
    Args:
        model_path (str): Path to the best model file.
        metrics (dict): Metrics of the best model.
    """
    setup_mlflow()
    with mlflow.start_run(run_name="Best_Model_Logging"):
        # Log numeric metrics to MLflow
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
            else:
                logger.warning(f"Skipping non-numeric metric {metric_name}: {metric_value}")

        # Log and register the best model in MLflow
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
            mlflow.xgboost.log_model(model, artifact_path="best_model")

            # Register the model
            model_name = "best_weather_model"
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
            mlflow.register_model(model_uri=model_uri, name=model_name)

            # Transition the model to the "Staging" stage
            client = MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Staging"
            )

        logger.info(f"Logged and registered the best model '{model_name}' in MLflow and transitioned to 'Staging' stage.")

def compare_and_select_best_model(metrics_paths):
    """
    Compare models from different training sessions and update the best model if necessary.
    Args:
        metrics_paths (list): List of paths to metrics JSON files for each model.
    Returns:
        tuple: (Path to best_model.json file, best model metrics path, best model file path).
    """
    if not metrics_paths:
        logger.error("No metrics files provided for comparison. Exiting.")
        return None

    best_metrics = None
    best_model_path = None

    # Iterate through all metrics files to find the best model
    for metrics_file in metrics_paths:
        metrics = load_metrics(metrics_file)
        if metrics is None:
            logger.warning(f"Skipping metrics file {metrics_file} due to loading issues.")
            continue

        # Compare based on RMSE (or other desired metric)
        if best_metrics is None or metrics["RMSE"] < best_metrics["RMSE"]:
            best_metrics = metrics
            model_file = metrics_file.replace("_metrics.json", ".pkl")
            best_model_path = model_file
            logger.info(f"New best model found with RMSE {metrics['RMSE']} from {metrics_file}")

    if best_model_path is None:
        logger.error("No valid models found during comparison.")
        return None

    # Path to store the best model metadata
    best_model_json_path = os.path.join(TEMP_DIR, "best_model.json")

    # Write the best model metadata to a JSON file
    try:
        best_model_data = {
            "best_model_path": best_model_path,
            "best_metrics": best_metrics,
        }
        with open(best_model_json_path, "w") as f:
            json.dump(best_model_data, f, indent=4)
        logger.info(f"Best model metadata saved to {best_model_json_path}")

        # Log the best model to MLflow
        try:
            flattened_metrics = simplify_metrics(best_metrics)
            log_best_model_to_mlflow(best_model_path, flattened_metrics)
        except Exception as e:
            logger.error(f"Failed to log the best model to MLflow: {e}")

        return best_model_json_path, best_metrics, best_model_path

    except Exception as e:
        logger.error(f"Failed to save best model metadata: {e}")
        return None
