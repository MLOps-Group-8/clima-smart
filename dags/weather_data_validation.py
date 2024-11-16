import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom test functions

def test_no_nulls(data, dataset_name):
    """Check for null values in the dataset."""
    if data.isnull().sum().sum() > 0:
        logging.error(f"{dataset_name} contains null values.")
    else:
        logging.info(f"No null values found in {dataset_name}.")

def test_positive_temperatures(data, dataset_name):
    """Ensure temperature columns have realistic values."""
    if 'temperature_2m_min' in data.columns:
        if not (data['temperature_2m_min'] >= -100).all():
            logging.error(f"{dataset_name} has unrealistic minimum temperatures.")
        else:
            logging.info(f"All minimum temperatures in {dataset_name} are realistic.")

    if 'temperature_2m_max' in data.columns:
        if not (data['temperature_2m_max'] <= 150).all():
            logging.error(f"{dataset_name} has unrealistic maximum temperatures.")
        else:
            logging.info(f"All maximum temperatures in {dataset_name} are realistic.")

def test_precipitation_non_negative(data, dataset_name):
    """Ensure precipitation values are non-negative."""
    if 'precipitation_sum' in data.columns:
        if not (data['precipitation_sum'] >= 0).all():
            logging.error(f"{dataset_name} has negative precipitation values.")
        else:
            logging.info(f"All precipitation values in {dataset_name} are non-negative.")

def test_schema_similarity(schema1, schema2):
    """Check if two schemas are similar by comparing feature keys and types."""
    schema1_features = schema1.get('features', {})
    schema2_features = schema2.get('features', {})

    if schema1_features != schema2_features:
        logging.warning("Schemas for daily and hourly data do not match.")
    else:
        logging.info("Schemas for daily and hourly data match.")


def validate_daily_weather_data(daily_data):
    """Run custom data validation checks on daily data."""

    # Run validation checks on daily data
    logging.info("Running validation checks on daily data.")
    test_no_nulls(daily_data, "Daily Data")
    test_positive_temperatures(daily_data, "Daily Data")
    test_precipitation_non_negative(daily_data, "Daily Data")

    logging.info("Data validation completed.")

def validate_hourly_weather_data(hourly_data):
    """Run custom data validation checks on hourly data."""
   
    # Run validation checks on hourly data
    logging.info("Running validation checks on hourly data.")
    test_no_nulls(hourly_data, "Hourly Data")
    test_positive_temperatures(hourly_data, "Hourly Data")
    test_precipitation_non_negative(hourly_data, "Hourly Data")

    logging.info("Data validation completed.")



def test_daily_data_quality_and_schema(daily_schema):
    """Test data quality by loading schemas from GCS and running tests."""
   
    # Log types for debugging
    logging.info(f"Type of daily_schema: {type(daily_schema)}")
    
    # Ensure schemas are dictionaries
    if not isinstance(daily_schema, dict):
        logging.error("Schemas are not in the expected dictionary format. Exiting validation.")
        return
    

def test_hourly_data_quality_and_schema(hourly_schema):
    """Test data quality by loading schemas from GCS and running tests."""
   
    # Log types for debugging
    logging.info(f"Type of hourly_schema: {type(hourly_schema)}")
    
    # Ensure schemas are dictionaries
    if not isinstance(hourly_schema, dict):
        logging.error("Schemas are not in the expected dictionary format. Exiting validation.")
        return
