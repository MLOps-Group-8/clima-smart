import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging
 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
# Define RMSE calculation
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE)."""
    return mean_squared_error(y_true, y_pred, squared=False)
 
# Perform slicing and calculate metrics for specific features
def calculate_metrics_for_features(data, features, models, targets, scalers, threshold_ratio=0.1):
    """
    Calculate RMSE metrics for two specific features and compare bias.
 
    Args:
        data (pd.DataFrame): The data containing features and targets.
        features (list): List of two features to compare for bias detection.
        models (dict): Dictionary of trained models for each target.
        targets (list): List of target variables.
        scalers (dict): Dictionary of scalers for each target.
        threshold_ratio (float): Bias threshold ratio.
 
    Returns:
        dict: Metrics and potential bias interpretations.
    """
    if len(features) != 2:
        raise ValueError("Exactly two features must be specified for bias detection.")
 
    metrics = {}
    interpretations = []
 
    for target in targets:
        logging.info(f"Calculating metrics for target: {target}")
        model = models[target]
        scaler_target = scalers[target]
 
        # Generate predictions for each feature
        for feature in features:
            unique_values = data[feature].unique()
            for value in unique_values:
                slice_data = data[data[feature] == value]
                if slice_data.empty:
                    logging.warning(f"No data points for slice '{value}' in feature '{feature}'")
                    continue
 
                X_slice = slice_data[features].values
                y_slice = slice_data[target].values
 
                # Convert X_slice to DMatrix and make predictions
                dmatrix_slice = xgb.DMatrix(X_slice)
                y_slice_pred = model.predict(dmatrix_slice)
 
                # Inverse transform the predictions and actual values
                y_slice_actual = scaler_target.inverse_transform(y_slice.reshape(-1, 1)).flatten()
                y_slice_pred_actual = scaler_target.inverse_transform(y_slice_pred.reshape(-1, 1)).flatten()
 
                # Calculate RMSE
                rmse = calculate_rmse(y_slice_actual, y_slice_pred_actual)
                metrics[f"{target}_{feature}_{value}"] = rmse
 
        # Bias detection
        average_rmse = np.mean(list(metrics.values()))
        bias_threshold = threshold_ratio * average_rmse
 
        for slice_name, rmse in metrics.items():
            if abs(rmse - average_rmse) > bias_threshold:
                interpretation = (
                    f"Potential bias detected in target '{target}', "
                    f"slice '{slice_name}': RMSE = {rmse:.4f} (Avg RMSE = {average_rmse:.4f})."
                )
                logging.info(interpretation)
                interpretations.append(interpretation)
 
    return {"metrics": metrics, "interpretations": interpretations}
 
# Add a binning column to the data
def bin_column(data, column_name, bins=4, labels=None):
    """
    Bin a continuous column into discrete categories.
 
    Args:
        data (pd.DataFrame): DataFrame containing the column to bin.
        column_name (str): The column to bin.
        bins (int): Number of bins to create.
        labels (list): Optional list of labels for each bin.
 
    Returns:
        pd.Series: Binned column as categorical data.
    """
    if labels is None:
        labels = [f"bin_{i}" for i in range(1, bins+1)]
    return pd.qcut(data[column_name], q=bins, labels=labels)
 
# Analyze the distribution of bins in the training data
def analyze_bin_distribution(data, column_name, bins=4, labels=None):
    """
    Analyze the distribution of bins for a specific column in the data.
 
    Args:
        data (pd.DataFrame): DataFrame with the column to analyze.
        column_name (str): The column to bin and analyze.
        bins (int): Number of bins.
        labels (list): Optional list of labels for the bins.
 
    Returns:
        pd.Series: Distribution of values in each bin.
    """
    data['bin_column'] = bin_column(data, column_name, bins=bins, labels=labels)
    bin_distribution = data['bin_column'].value_counts()
    logging.info(f"Distribution of bins for {column_name}:\n{bin_distribution}")
    return bin_distribution