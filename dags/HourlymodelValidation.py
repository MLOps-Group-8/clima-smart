import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 1: Validate each target variable
for target in targets:
    print(f"\nValidating model for target: {target}")

    # Make predictions
    y_val_pred = model.predict(xgb.DMatrix(X_val))
    y_test_pred = model.predict(xgb.DMatrix(X_test))

    # Fit the scaler on the entire target data and then use it for inverse transformation
    scaler_target = StandardScaler()
    scaler_target.fit(data[[target]])  # Fit on the entire dataset for this target

    # Inverse transform the test and validation predictions
    y_val_actual = scaler_target.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_val_pred_actual = scaler_target.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
    y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_actual = scaler_target.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # Metrics function to evaluate the model
    def evaluate_model(y_true, y_pred, label="Set"):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{label} Metrics: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Evaluate on validation and test sets
    evaluate_model(y_val_actual, y_val_pred_actual, label=f"Validation Set ({target})")
    evaluate_model(y_test_actual, y_test_pred_actual, label=f"Test Set ({target})")

    # Step 2: Plot residuals (difference between actual and predicted values)
    residuals = y_test_actual - y_test_pred_actual
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"Residual Distribution ({target})")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    # Step 3: Plot Actual vs Predicted values for each target variable
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_actual, y_test_pred_actual, alpha=0.6)
    plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
    plt.title(f"Actual vs Predicted: {target}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()