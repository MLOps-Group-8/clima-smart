import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import xgboost as xgb
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(data, features, targets):
    """
    Process the data for model training.
    """
    logging.info("Processing data for model training")
    
    # Fill missing values
    data.ffill(inplace=True)

    # Normalize the features
    scaler_features = StandardScaler()
    data[features] = scaler_features.fit_transform(data[features])

    processed_data = {}

    for target in targets:
        # No target scaling; use raw target values
        # Split the data into features (X) and target (y)
        X = data[features]
        y = data[target]

        # Train/Validation/Test Split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        processed_data[target] = {
            "X_train": X_train,
            "X_val": X_val,  # Features only
            "X_test": X_test,  # Features only
            "y_train": y_train,  # Targets
            "y_val": y_val,  # Targets
            "y_test": y_test,  # Targets
        }

    logging.info("Data processing completed")
    return processed_data, scaler_features, None

def train_model(X_train, X_val, y_train, y_val, params):
    """
    Train an XGBoost model with early stopping.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    return model

def hyperparameter_tuning(data, space, max_evals=10):
    """
    Perform hyperparameter tuning using Hyperopt.
    """
    def objective(params):
        params["max_depth"] = int(params["max_depth"])
        params["objective"] = "reg:squarederror"
        params["seed"] = 42

        model = train_model(
            data["X_train"], data["X_val"], data["y_train"], data["y_val"], params
        )
        dval = xgb.DMatrix(data["X_val"])
        y_pred = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(data["y_val"], y_pred))
        return {"loss": rmse, "status": STATUS_OK, "model": model}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    best_model_index = np.argmin([trial["result"]["loss"] for trial in trials.trials])
    best_model = trials.trials[best_model_index]["result"]["model"]
    return best_model, best

def run_model_training(data_path, output_file, task_instance):
    """
    Main function to process data, perform hyperparameter tuning, and save individual models
    as well as a combined file with all models.

    Args:
        data_path (str): Path to the input data file.
        output_file (str): Path for the combined all_models.pkl file.
        task_instance: Airflow TaskInstance object to push XCom values.

    Returns:
        str: Path to the combined models file (all_models.pkl).
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    features = [
        'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 'rain_sum',
        'showers_sum', 'daylight_duration', 'precipitation_sum', 'temperature_range',
        'diurnal_temp_range', 'precipitation_intensity'
    ]
    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum']

    # Process the data
    processed_data, _, _ = process_data(data, features, targets)

    space = {
        'max_depth': hp.quniform('max_depth', 3, 5, 1),
        'learning_rate': hp.uniform('learning_rate', 0.05, 0.1),
        'subsample': hp.uniform('subsample', 0.7, 0.8),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 0.8),
    }

    # Dictionary to store all trained models
    all_models = {}

    # Directory to save individual target models
    target_models_dir = os.path.join(os.path.dirname(output_file), "target_models")
    os.makedirs(target_models_dir, exist_ok=True)

    for target, splits in processed_data.items():
        # Train the best model for the target
        best_model, best_params = hyperparameter_tuning(splits, space)
        
        # Save the best model for the target as an individual .pkl file
        target_model_path = os.path.join(target_models_dir, f"{target}_model.pkl")
        with open(target_model_path, "wb") as f:
            pickle.dump({"model": best_model, "best_params": best_params}, f)
        
        logging.info(f"Best model for {target} saved at {target_model_path}")

        # Push the individual model path to XCom for the specific target
        task_instance.xcom_push(key=f"{target}_model_path", value=target_model_path)

        # Add the model and its parameters to the all_models dictionary
        all_models[target] = {"model": best_model, "best_params": best_params}

    # Save all models together in a single .pkl file
    with open(output_file, "wb") as f:
        pickle.dump(all_models, f)

    logging.info(f"All models saved together at {output_file}")
    return output_file