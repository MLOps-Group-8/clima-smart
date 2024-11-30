import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def bias_detection(data, models, features, targets, scalers, slice_column='temperature_2m'):
    # Create temperature bins
    data['temp_bin'] = pd.qcut(data[slice_column], q=4, labels=['low', 'medium-low', 'medium-high', 'high'])

    bias_metrics = {}

    for target in targets:
        model = models[target]
        scaler_target = scalers[target]

        for bin_label in data['temp_bin'].unique():
            slice_data = data[data['temp_bin'] == bin_label]

            X_slice = slice_data[features]
            y_slice = slice_data[target]

            # Make predictions
            dmatrix_slice = xgb.DMatrix(X_slice)
            y_pred = model.predict(dmatrix_slice)

            # Inverse transform predictions and actual values
            y_slice_actual = scaler_target.inverse_transform(y_slice.values.reshape(-1, 1)).flatten()
            y_pred_actual = scaler_target.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            # Calculate RMSE
            rmse = calculate_rmse(y_slice_actual, y_pred_actual)
            bias_metrics[f"{target}_{bin_label}"] = rmse

    return bias_metrics

# Assuming 'data' is your DataFrame, 'models' is your dictionary of trained models,
# 'features' is your list of feature columns, 'targets' is your list of target variables,
# and 'scalers' is your dictionary of fitted StandardScalers for each target

# Perform bias detection
bias_metrics = bias_detection(data, models, features, targets, scalers)

# Calculate average RMSE across all slices
average_rmse = np.mean(list(bias_metrics.values()))

# Set a threshold for bias detection (e.g., 20% deviation from average)
bias_threshold = 0.2 * average_rmse

print("Bias Detection Results:")
for slice_name, rmse in bias_metrics.items():
    if abs(rmse - average_rmse) > bias_threshold:
        print(f"Potential bias detected in {slice_name}: RMSE = {rmse:.4f} (Avg RMSE = {average_rmse:.4f})")

# Analyze distribution of temperature bins
print("\nDistribution of temperature bins:")
print(data['temp_bin'].value_counts(normalize=True))

# Feature correlation analysis
print("\nFeature Correlation Analysis:")
correlation_matrix = data[features + targets].corr()

# Plot heatmap of feature correlations
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Print correlation with target variables
for target in targets:
    print(f"\nCorrelations with {target}:")
    correlations = correlation_matrix[target].sort_values(ascending=False)
    print(correlations[correlations.index != target])

# Analyze feature importance for each target
for target in targets:
    print(f"\nFeature Importance for {target}:")
    feature_importance = models[target].get_score(importance_type='gain')
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:5]:  # Top 5 features
        print(f"{feature}: {importance:.4f}")

# Analyze performance across all bins for each target
print("\nPerformance across all bins:")
for target in targets:
    print(f"\nTarget: {target}")
    for bin_label in data['temp_bin'].unique():
        rmse = bias_metrics[f"{target}_{bin_label}"]
        print(f"  {bin_label}: RMSE = {rmse:.4f}")