import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Select relevant columns (features and multiple targets)
features = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
    'cloud_cover', 'pressure_msl', 'wind_speed_10m', 'wind_direction_10m',
    'is_day', 'hour', 'is_weekend', 'month', 'is_holiday'
]
targets = ['snowfall', 'rain', 'apparent_temperature', 'wind_gusts_10m']

# Step 3: Handle missing values
data.dropna(subset=targets + features, inplace=True)

# Step 4: Normalize the features
scaler_features = StandardScaler()
data[features] = scaler_features.fit_transform(data[features])

# Initialize results dictionary
results = {}

# Step 5: Iterate through each target variable
for target in targets:
    print(f"Training model for target: {target}")

    # Normalize the target variable
    scaler_target = StandardScaler()
    data[target] = scaler_target.fit_transform(data[[target]])

    # Split the data into features (X) and target (y)
    X = data[features].values
    y = data[target].values

    # Train/Validation/Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert the data into DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define model parameters
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

    val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_r2 = r2_score(y_test, y_test_pred)

    # Store metrics and model
    results[target] = {
        "Validation RMSE": val_rmse,
        "Test RMSE": test_rmse,
        "Test R2 Score": test_r2,
        "Model": model,
        "Scaler": scaler_target,
        "X_test": X_test,
    }

    # SHAP Feature Importance
    print(f"Computing SHAP values for target: {target}")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Summary Plot
    print(f"Generating SHAP summary plot for {target}")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance for {target}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.title(f"SHAP Summary Plot for {target}")
    plt.tight_layout()
    plt.show()

# Step 6: Display the results
results_df = pd.DataFrame({k: {m: v[m] for m in ['Validation RMSE', 'Test RMSE', 'Test R2 Score']} for k, v in results.items()}).T
print("\nModel Performance Metrics for All Targets:")
print(results_df)