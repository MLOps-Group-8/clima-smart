import logging
from google.cloud import storage
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_gcs(bucket_name, file_path):
    """
    Load engineered data from a GCS bucket.

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
    df = pd.read_csv(BytesIO(data))
    logging.info("Data successfully loaded from GCS")
    return df

def process_data(data):
    """
    Process the loaded data for model training.

    Args:
        data (pd.DataFrame): Raw data.

    Returns:
        Tuple: Preprocessed training and test data (X_train, X_test, y_train, y_test).
    """
    logging.info("Processing data for model training")
    data_cleaned = data.dropna()
    X = data_cleaned.drop(columns=[
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "precipitation", "rain", "snowfall", "wind_speed_10m",
        "wind_direction_10m", "surface_pressure", "apparent_temperature"
    ])
    X = X.drop(columns=["temp_rolling_mean_24h"], errors='ignore')
    X['datetime'] = pd.to_datetime(X['datetime'], errors='coerce')
    X['year'] = X['datetime'].dt.year
    X['month'] = X['datetime'].dt.month
    X['day'] = X['datetime'].dt.day
    X['hour'] = X['datetime'].dt.hour
    X = X.drop(columns=["datetime"])
    X = pd.get_dummies(X, columns=['wind_category'], drop_first=True)
    y = data_cleaned["apparent_temperature"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.expand_dims(X_scaled, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    logging.info("Data processing completed")
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    """
    Build and compile the LSTM model.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        model: Compiled LSTM model.
    """
    logging.info("Building LSTM model")
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    logging.info("Model built and compiled successfully")
    return model

def train_model(X_train, X_test, y_train, y_test):
    """
    Train the LSTM model.

    Args:
        X_train, X_test, y_train, y_test: Preprocessed data.

    Returns:
        model: Trained LSTM model.
    """
    logging.info("Starting model training")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    logging.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Args:
        model: Trained model.
        X_test, y_test: Test data.

    Returns:
        None
    """
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Evaluation completed. RMSE: {rmse}, R^2: {r2}")
    return rmse, r2

def save_model(model, model_path="lstm_model.json", weights_path="lstm_model_weights.h5"):
    """
    Save the trained model.

    Args:
        model: Trained model.
        model_path (str): Path to save the model architecture.
        weights_path (str): Path to save the model weights.

    Returns:
        None
    """
    logging.info(f"Saving model to {model_path} and weights to {weights_path}")
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_path)
    logging.info("Model and weights saved successfully")
