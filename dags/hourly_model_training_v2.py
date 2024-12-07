import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_and_save_models(data_path, model_dir, target_features):
    """
    Train XGBoost models for hourly data and save them.
    """
    logging.info(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    date_features = ['hour', 'month', 'day_of_year', 'week_of_year', 'is_weekend']

    os.makedirs(model_dir, exist_ok=True)
    models = {}
    for target in target_features:
        logging.info(f"Training model for {target}...")
        X = data[date_features]
        y = data[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train XGBoost model
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"{target} - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

        # Save model
        model_path = os.path.join(model_dir, f"{target}_model.json")
        model.save_model(model_path)
        logging.info(f"Saved model for {target} at {model_path}")
        models[target] = model

    return models

def monitor_model_performance(models, data_path, target_features, thresholds):
    """
    Monitor model performance for hourly data.
    """
    logging.info(f"Loading data from {data_path} for monitoring...")
    data = pd.read_csv(data_path)
    date_features = ['hour', 'month', 'day_of_year', 'week_of_year', 'is_weekend']

    performance_metrics = {}
    retrain_needed = False
    for target in target_features:
        logging.info(f"Evaluating model for {target}...")
        X = data[date_features]
        y_actual = data[target]
        y_pred = models[target].predict(X)

        # Calculate metrics
        rmse = mean_squared_error(y_actual, y_pred, squared=False)
        r2 = r2_score(y_actual, y_pred)
        performance_metrics[target] = {'rmse': rmse, 'r2': r2}

        # Log metrics
        logging.info(f"{target} - RMSE: {rmse}, R²: {r2}")

        # Check thresholds
        if rmse > thresholds['rmse'] or r2 < thresholds['r2']:
            logging.warning(f"{target} metrics exceeded thresholds. RMSE: {rmse}, R²: {r2}")
            retrain_needed = True

    return performance_metrics, retrain_needed

def load_models(model_dir, target_features):
    """
    Load trained models for each target feature.
    """
    logging.info(f"Loading models from {model_dir}...")
    models = {}
    for target in target_features:
        model_path = os.path.join(model_dir, f"{target}_model.json")
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        models[target] = model
        logging.info(f"Loaded model for {target} from {model_path}")
    return models
