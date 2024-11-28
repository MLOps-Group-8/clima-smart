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

def load_data_from_gcs(bucket_name, file_path):
    """
    Load engineered daily data from a GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        file_path (str): The path to the file in the GCS bucket.

    Returns:
        pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    logging.info(f"Loading data from GCS bucket: {bucket_name}, file path: {file_path}")
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    logging.info("Data successfully loaded from GCS")
    return df

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
    target_scalers = {}

    for target in targets:
        # Normalize the target variable
        scaler_target = StandardScaler()
        data[target] = scaler_target.fit_transform(data[[target]])

        # Split the data into features (X) and target (y)
        X = data[features]
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

<<<<<<< Updated upstream:dags/dailymodeltraining.py
    logging.info(f"Evaluation completed. RMSE: {rmse}, R^2: {r2}")
    return {"RMSE": rmse, "R2": r2}

def save_model_to_gcs(model, bucket_name, file_name):
    """
    Save a trained model as a pickle file in Google Cloud Storage.

    Args:
        model: The trained model to save.
        bucket_name (str): Name of the GCS bucket.
        file_name (str): Path to save the model in the GCS bucket.

    Returns:
        None
    """
    logging.info(f"Saving model to GCS bucket {bucket_name} at {file_name}")

    # Initialize the GCS client and bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Serialize the model into a pickle object
    output = io.BytesIO()
    pickle.dump(model, output)
    output.seek(0)

    # Upload the pickle object to GCS
    blob.upload_from_file(output, content_type='application/octet-stream')
    logging.info(f"Model successfully saved to GCS: gs://{bucket_name}/{file_name}")
=======
    best_model_index = np.argmin([trial["result"]["loss"] for trial in trials.trials])
    best_model = trials.trials[best_model_index]["result"]["model"]
    return best_model, best

def run_model_training(data_path, output_file):
    """
    Main function to process data, perform hyperparameter tuning, and save the best models.
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    features = [
        'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_min', 'rain_sum',
        'showers_sum', 'daylight_duration', 'precipitation_sum', 'temperature_range',
        'diurnal_temp_range', 'precipitation_intensity'
    ]
    targets = ['apparent_temperature_max', 'sunshine_duration', 'snowfall_sum']

    processed_data, _, _ = process_data(data, features, targets)

    space = {
        'max_depth': hp.quniform('max_depth', 3, 5, 1),
        'learning_rate': hp.uniform('learning_rate', 0.05, 0.1),
        'subsample': hp.uniform('subsample', 0.7, 0.8),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 0.8),
    }

    # Dictionary to store all trained models
    all_models = {}

    for target, splits in processed_data.items():
        best_model, best_params = hyperparameter_tuning(splits, space)
        all_models[target] = {"model": best_model, "best_params": best_params}

        logging.info(f"Best model for {target} trained successfully")

    # Save all models together in a single .pkl file
    with open(output_file, "wb") as f:
        pickle.dump(all_models, f)
    
    logging.info(f"All models saved together at {output_file}")
    return output_file
>>>>>>> Stashed changes:dags/daily_model_training.py
