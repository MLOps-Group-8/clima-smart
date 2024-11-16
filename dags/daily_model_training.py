import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from google.cloud import storage
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(data, features, targets):
    """
    Process the data for model training.

    Args:
        data (pd.DataFrame): Raw data.
        features (list): List of feature column names.
        targets (list): List of target column names.

    Returns:
        dict: Processed data split for each target variable and scalers.
    """
    logging.info("Processing data for model training")

    # Fill missing values
    data.ffill(inplace=True)

    # Normalize the features
    scaler_features = StandardScaler()
    data[features] = scaler_features.fit_transform(data[features])

    processed_data = {}
    target_scalers = {}

    for target in targets:
        # Normalize the target variable
        scaler_target = StandardScaler()
        data[target] = scaler_target.fit_transform(data[[target]])

        # Split the data into features (X) and target (y)
        X = data[features]  # Retain as DataFrame
        y = data[target].values

        # Train/Validation/Test Split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        processed_data[target] = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }
        target_scalers[target] = scaler_target

    logging.info("Data processing completed")
    return processed_data, scaler_features, target_scalers

def train_model(X_train, X_val, y_train, y_val, params=None):
    """
    Train an XGBoost model with early stopping.

    Args:
        X_train, X_val: Feature matrices for training and validation.
        y_train, y_val: Target vectors for training and validation.
        params (dict): Hyperparameters for the XGBoost model.

    Returns:
        model: Trained XGBoost model.
    """
    logging.info("Starting model training")

    # Default parameters
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42
        }

    # Convert data into DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train the model with early stopping
    evals = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    logging.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test, scaler_target):
    """
    Evaluate the trained model.

    Args:
        model: Trained model.
        X_test, y_test: Test data.
        scaler_target: Scaler for inverse-transforming target values.

    Returns:
        dict: Evaluation metrics (RMSE, R^2).
    """
    logging.info("Evaluating model")
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    # Inverse-transform predictions and actual target values
    y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    y_pred_actual = scaler_target.inverse_transform(y_pred.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)

    logging.info(f"Evaluation completed. RMSE: {rmse}, R^2: {r2}")
    return {"RMSE": rmse, "R2": r2}