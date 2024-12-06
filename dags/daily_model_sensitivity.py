import shap
import matplotlib.pyplot as plt
import logging
import os
from utils import save_plot_to_gcs
import xgboost as xgb


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def perform_feature_importance_analysis(model, X_test, feature_names, target, bucket_name):
    """
    Perform feature importance analysis using SHAP for the given model and data.

    Args:
        model (xgb.Booster): Trained XGBoost model.
        X_test (pd.DataFrame): Test dataset features.
        feature_names (list): List of feature names.
        target (str): Target variable name.
        bucket_name (str): Name of the GCS bucket to save the plots.
    """
    logging.info(f"Starting feature importance analysis for target: {target}")

    # Convert test data to DMatrix for SHAP analysis
    dmatrix_test = xgb.DMatrix(X_test)

    # Use SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Generate SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    shap_plot_name = f"{target}_shap_summary_plot"
    save_plot_to_gcs(bucket_name, shap_plot_name)
    plt.close()  # Close the sensitivity plot

    logging.info(f"SHAP summary plot for {target} saved as {shap_plot_name}.png in GCS.")

    # Generate SHAP dependence plots for each feature
    for feature in feature_names:
        plt.figure()
        shap.dependence_plot(feature, shap_values, X_test, feature_names=feature_names, show=False)
        dependence_plot_name = f"{target}_shap_dependence_plot_{feature}"
        save_plot_to_gcs(bucket_name, dependence_plot_name)
        plt.close()  # Close the sensitivity plot
        logging.info(f"SHAP dependence plot for {feature} saved as {dependence_plot_name}.png in GCS.")
    return None

def perform_hyperparameter_sensitivity_analysis(trials, target, bucket_name):
    """
    Perform hyperparameter sensitivity analysis using the hyperopt trials.

    Args:
        trials (hyperopt.Trials): Hyperopt trials object from hyperparameter tuning.
        target (str): Target variable name.
        bucket_name (str): Name of the GCS bucket to save the plots.
    """
    logging.info(f"Starting hyperparameter sensitivity analysis for target: {target}")

    results = trials.results
    hyperparams = trials.vals

    # Collect RMSE and hyperparameters
    rmse_list = [result["loss"] for result in results]
    param_names = list(hyperparams.keys())

    # Plot the sensitivity of each hyperparameter
    for param in param_names:
        plt.figure()
        plt.scatter(hyperparams[param], rmse_list)
        plt.xlabel(param)
        plt.ylabel("RMSE")
        plt.title(f"Sensitivity of {param} on RMSE for {target}")
        plot_name = f"{target}_hyperparameter_sensitivity_{param}"
        plt.close()  # Close the sensitivity plot
        save_plot_to_gcs(bucket_name, plot_name)
        logging.info(f"Hyperparameter sensitivity plot for {param} saved as {plot_name}.png in GCS.")
    return None
