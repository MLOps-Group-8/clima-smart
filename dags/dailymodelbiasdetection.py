import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define RMSE calculation
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE)."""
    return mean_squared_error(y_true, y_pred, squared=False)

def bin_features(data, column_name, bins=4, labels=None):
    """
    Bin a continuous column into discrete categories.

    Args:
        data (pd.DataFrame): DataFrame containing the column to bin.
        column_name (str): The column to bin.
        bins (int): Number of bins to create.
        labels (list): Optional labels for the bins.

    Returns:
        pd.Series: Binned column as categorical data.
    """
    if labels is None:
        labels = [f"bin_{i}" for i in range(1, bins + 1)]
    return pd.cut(data[column_name], bins=bins, labels=labels)


# Perform slicing and calculate metrics for specific features
def calculate_metrics_for_features(data, slicing_features, models, targets, scalers, threshold_ratio=0.1):
    """
    Calculate RMSE metrics for slicing features and compare bias.

    Args:
        data (pd.DataFrame): The data containing features and targets.
        slicing_features (list): List of features to slice for bias detection.
        models (dict): Dictionary of trained models for each target.
        targets (list): List of target variables.
        scalers (dict): Dictionary of scalers for each target.
        threshold_ratio (float): Bias threshold ratio.

    Returns:
        dict: Metrics and potential bias interpretations.
    """
    metrics = {}
    interpretations = []

    for target in targets:
        logging.info(f"Calculating metrics for target: {target}")
        model = models[target]
        scaler_target = scalers[target]

        for feature in slicing_features:
            if feature not in data.columns:
                logging.warning(f"Slicing feature '{feature}' not found in data. Skipping.")
                continue

            unique_values = data[feature].unique()
            for value in unique_values:
                slice_data = data[data[feature] == value]
                if slice_data.empty:
                    logging.warning(f"No data points for slice '{value}' in slicing feature '{feature}'")
                    continue

                # Log available columns in the slice
                logging.info(f"Columns in slice_data for '{feature} = {value}': {list(slice_data.columns)}")

                # Select input features for the model by excluding targets and slicing features
                drop_columns = [col for col in targets + slicing_features if col in slice_data.columns]
                X_slice = slice_data.drop(columns=drop_columns).values

                if target in slice_data.columns:
                    y_slice = slice_data[target].values
                else:
                    logging.warning(f"Target '{target}' not found in slice_data columns for '{feature} = {value}'. Skipping.")
                    continue

                # Convert X_slice to DMatrix and make predictions
                dmatrix_slice = xgb.DMatrix(X_slice)
                y_slice_pred = model.predict(dmatrix_slice)

                # Handle scalers for inverse transformation
                if hasattr(scaler_target, 'inverse_transform'):
                    y_slice_actual = scaler_target.inverse_transform(y_slice.reshape(-1, 1)).flatten()
                    y_slice_pred_actual = scaler_target.inverse_transform(y_slice_pred.reshape(-1, 1)).flatten()
                else:
                    y_slice_actual = y_slice
                    y_slice_pred_actual = y_slice_pred

                # Calculate RMSE
                rmse = calculate_rmse(y_slice_actual, y_slice_pred_actual)
                metrics[f"{target}_{feature}_{value}"] = rmse
                logging.info(f"Calculated RMSE for {target}, {feature}={value}: {rmse}")

        # Bias detection
        if metrics:
            average_rmse = np.mean(list(metrics.values()))
            bias_threshold = threshold_ratio * average_rmse

            for slice_name, rmse in metrics.items():
                if abs(rmse - average_rmse) > bias_threshold:
                    interpretation = (
                        f"Potential bias detected in target '{target}', "
                        f"slice '{slice_name}': RMSE = {rmse:.4f} (Avg RMSE = {average_rmse:.4f})."
                    )
                    interpretations.append(interpretation)

    # Summarize results
    logging.info("Bias detection completed. Summary of interpretations:")
    for interpretation in interpretations:
        logging.info(interpretation)

    return {"metrics": metrics, "interpretations": interpretations}
