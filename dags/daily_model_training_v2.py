import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_and_save_models(data_path, model_dir, target_features):
    """
    Train XGBoost models for multiple target features and save them.
    """
    logging.info(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    # Feature engineering
    data['month'] = data['date'].dt.month
    data['day_of_year'] = data['date'].dt.day_of_year
    data['week_of_year'] = data['date'].dt.isocalendar().week
    data['is_weekend'] = data['date'].dt.weekday >= 5
    date_features = ['month', 'day_of_year', 'week_of_year', 'is_weekend']
    
    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    models = {}
    for target in target_features:
        logging.info(f"Training model for {target}...")
        X = data[date_features]
        y = data[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred, squared=False)
        logging.info(f"{target} - RMSE: {rmse}")
        
        # Save model
        model_path = os.path.join(model_dir, f"{target}_model.json")
        model.save_model(model_path)
        logging.info(f"Model for {target} saved at {model_path}")
        models[target] = model
    
    return models

def predict_features(models, date, target_features):
    """
    Predict multiple features based on the date.
    """
    # Create date-based features
    input_data = pd.DataFrame({'date': [date]})
    input_data['date'] = pd.to_datetime(input_data['date'])
    input_data['month'] = input_data['date'].dt.month
    input_data['day_of_year'] = input_data['date'].dt.day_of_year
    input_data['week_of_year'] = input_data['date'].dt.isocalendar().week
    input_data['is_weekend'] = input_data['date'].dt.weekday >= 5
    input_features = input_data[['month', 'day_of_year', 'week_of_year', 'is_weekend']]
    
    # Predict for each target feature
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(input_features)[0]
    
    return predictions

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
