import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from fairlearn.metrics import MetricFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
from constants import *
from utils import (save_plot_to_gcs, save_object_to_gcs)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Slicing Function
def create_slicing_features(data, slicing_definitions):
    """
    Add slicing features to the dataset based on provided slicing definitions.

    Args:
        data (pd.DataFrame): The dataset.
        slicing_definitions (dict): Slicing criteria. Example:
            {'temperature_2m_bin': {'column': 'temperature_2m', 'bins': 4, 'labels': [...]}}
    
    Returns:
        pd.DataFrame: Dataset with slicing features encoded numerically.
    """
    for feature_name, params in slicing_definitions.items():
        column = params['column']
        bins = params['bins']
        labels = params.get('labels', None)
        data[feature_name] = pd.cut(data[column], bins=bins, labels=labels).astype(str)  # Convert to string
        
        # Convert categorical labels to numerical codes
        data[feature_name] = data[feature_name].astype('category').cat.codes
    return data

def evaluate_model_bias(X_test, y_test, model, slicing_features):
    """
    Evaluate model bias using slicing techniques and Fairlearn's MetricFrame.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test targets.
        model: Trained model.
        slicing_features (list): List of slicing features to evaluate.

    Returns:
        dict: Metrics and potential bias insights.
    """
    bucket_name = BUCKET_NAME
    
    # Extract the actual model from the dictionary
    model = model["model"]
    
    # Convert only the original features to DMatrix for predictions
    original_features = X_test.drop(columns=slicing_features).copy()
    dmatrix_test = xgb.DMatrix(original_features)

    # Generate predictions
    predictions = model.predict(dmatrix_test)

    # Store metrics for each slice
    metrics = {}

    for slice_feature in slicing_features:
        logging.info(f"Evaluating bias for slicing feature: {slice_feature}")
        
        # Ensure the slicing feature is valid
        if slice_feature not in X_test.columns or X_test[slice_feature].empty:
            logging.warning(f"Slicing feature '{slice_feature}' is missing or empty. Skipping.")
            continue

        # Use MetricFrame to evaluate RMSE per slice
        metric_frame = MetricFrame(
            metrics={'RMSE': lambda y, y_pred: mean_squared_error(y, y_pred, squared=False)},
            y_true=y_test,
            y_pred=predictions,
            sensitive_features=X_test[slice_feature]
        )
        
        # Extract scalar overall RMSE
        overall_rmse = metric_frame.overall.item() if hasattr(metric_frame.overall, "item") else metric_frame.overall
        metrics[slice_feature] = (overall_rmse, metric_frame.by_group)
        logging.info(f"Metrics for {slice_feature}: {metric_frame.by_group}")
        
        # Save metrics as a .pkl file to GCS
        metrics_data = {'overall': overall_rmse, 'by_group': metric_frame.by_group}
        metrics_path = f"assets/hourly_bias_analysis/{slice_feature}_metrics.pkl"
        save_object_to_gcs(bucket_name, metrics_data, metrics_path)
         
    return metrics


# Visualization Function
def plot_bias_metrics(metrics, slicing_features):
    """
    Plot bias metrics across slices.

    Args:
        metrics (dict): Bias metrics for each slicing feature.
        slicing_features (list): List of slicing features evaluated.
    """
    
    bucket_name = BUCKET_NAME
    
    for feature in slicing_features:
        overall, by_group = metrics[feature]
        
        # Ensure `overall` is a scalar
        overall = overall.item() if hasattr(overall, "item") else overall

        plt.figure(figsize=(10, 6))
        by_group.plot(kind='bar', color='skyblue')
        plt.axhline(y=overall, color='red', linestyle='--', label='Overall RMSE')
        plt.title(f"Bias Metrics for {feature}")
        plt.xlabel("Slices")
        plt.ylabel("RMSE")
        plt.legend()
        
        # Save the plot to GCS
        plot_name = f"bias_metrics_{feature}"
        save_plot_to_gcs(bucket_name, plot_name)
        plt.close()

# Example Workflow for Bias Detection
def run_bias_detection_workflow(data, features, target, model, slicing_definitions):
    """
    Full workflow to perform bias detection and evaluate metrics.

    Args:
        data (pd.DataFrame): Full dataset.
        features (list): Feature columns for model input.
        target (str): Target variable.
        model: Trained model.
        slicing_definitions (dict): Slicing configurations.

    Returns:
        dict: Bias metrics for the target variable.
    """
    # Step 1: Add slicing features
    logging.info("Adding slicing features to the dataset")
    data_with_slices = create_slicing_features(data, slicing_definitions)

    # Step 2: Train-Test Split
    X = data_with_slices[features + list(slicing_definitions.keys())]
    y = data_with_slices[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Evaluate Bias
    metrics = evaluate_model_bias(X_test, y_test, model, list(slicing_definitions.keys()))
    plot_bias_metrics(metrics, slicing_features=list(slicing_definitions.keys()))

    return metrics
