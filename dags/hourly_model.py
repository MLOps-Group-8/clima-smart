import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('weather_data_engineered_hourly_data.csv')

# Step 2: Select relevant columns (features and multiple targets)
features = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
    'cloud_cover', 'pressure_msl', 'wind_speed_10m', 'wind_direction_10m',
    'is_day', 'hour', 'is_weekend', 'month', 'is_holiday'
]

# Add new target variables: sunshine_duration and wind_gusts_10m
targets = ['snowfall', 'rain', 'apparent_temperature', 'wind_gusts_10m']

# Step 3: Handle missing values by dropping rows with missing data in features or targets
data = data.dropna(subset=targets + features)

# Step 4: Normalize the features
scaler_features = StandardScaler()
data[features] = scaler_features.fit_transform(data[features])

# Step 5: Iterate through each target variable
results = {}
for target in targets:
    # Normalize the target variable
    scaler_target = StandardScaler()
    data[target] = scaler_target.fit_transform(data[[target]])

    # Split the data into features (X) and target (y)
    X = data[features].values
    y = data[target].values

    # Train/Validation/Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert the data into DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define model parameters for XGBoost
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

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

    # Evaluate the model on validation and test sets
    y_val_pred = model.predict(dval)
    y_test_pred = model.predict(dtest)

    # Calculate evaluation metrics for validation and test sets
    val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Store the results in a dictionary for each target variable
    results[target] = {
        "Validation RMSE": val_rmse,
        "Test RMSE": test_rmse,
        "Test R2 Score": test_r2,
        "Test MAE": test_mae,
    }

    # Inverse transform the target values for comparison (actual vs predicted)
    y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred_actual = scaler_target.inverse_transform(y_test_pred.reshape(-1, 1))

    # Plot Actual vs Predicted values for each target variable
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_actual, y_test_pred_actual, alpha=0.6)
    plt.title(f"Predictions vs Actual: {target}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot([y_test_actual.min(), y_test_actual.max()],
             [y_test_actual.min(), y_test_actual.max()], 'r--')
    plt.show()

# Step 6: Display the results in a DataFrame format
results_df = pd.DataFrame(results).T
print("\nModel Performance Metrics for All Targets:")
print(results_df)