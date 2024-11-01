import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_session():
    """
    Configures and returns a session object with caching and retry mechanisms for robust API requests.

    Returns:
    openmeteo_requests.Client: A client configured with caching and retries.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)
    logging.info("Session setup with caching and retries.")
    return client
    
def fetch_hourly_weather_data(client, api_url, params):
    """
    Fetches hourly weather data using the provided client, API URL, and parameters.

    Args:
    client (openmeteo_requests.Client): The client to use for fetching data.
    api_url (str): The URL of the weather API.
    params (dict): Parameters for the API request.

    Returns:
    dict: The fetched data.
    """
    logging.info(f"Fetching hourly weather data from {api_url} with params {params}")
    params.update({"hourly": "long list of hourly parameters"})
    data = client.weather_api(api_url, params=params)
    logging.info("Hourly data fetch successful.")
    return data

def process_daily_weather_data(response):
    """
    Processes the raw API response for daily weather data into a structured pandas DataFrame.

    Args:
    response (list): The API response containing daily weather data.

    Returns:
    pandas.DataFrame: The processed daily weather data.
    """
    logging.info("Processing daily weather data.")
    daily_response = response[0]  # Assume response is a list and we need the first item
    daily = daily_response.Daily()
    daily_data = pd.DataFrame({
        # Assuming daily data keys are correctly mapped here
    })
    logging.info("Daily weather data processing complete.")
    return daily_data

def fetch_daily_weather_data(client, api_url, params):
    params.update({"daily": "weather_code,temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,daylight_duration,sunshine_duration,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration"})
    return client.weather_api(api_url, params=params)

def fetch_hourly_weather_data(client, api_url, params):
    params.update({"hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation,rain,snowfall,snow_depth,weather_code,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,et0_fao_evapotranspiration,vapour_pressure_deficit,wind_speed_10m,wind_speed_100m,wind_direction_10m,wind_direction_100m,wind_gusts_10m,soil_temperature_0_to_7cm,soil_temperature_7_to_28cm,soil_temperature_28_to_100cm,soil_temperature_100_to_255cm,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm"})
    return client.weather_api(api_url, params=params)

def process_daily_weather_data(response):
    daily_response = response[0] 
    daily = daily_response.Daily()
    daily_data = pd.DataFrame({
        "date": pd.date_range(start=pd.to_datetime(daily.Time(), unit="s", utc=True), end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=daily.Interval()), inclusive="left"),
        "weather_code": daily.Variables(0).ValuesAsNumpy(),
        "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
        "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
        "temperature_2m_mean": daily.Variables(3).ValuesAsNumpy(),
        "apparent_temperature_max": daily.Variables(4).ValuesAsNumpy(),
        "apparent_temperature_min": daily.Variables(5).ValuesAsNumpy(),
        "apparent_temperature_mean": daily.Variables(6).ValuesAsNumpy(),
        "daylight_duration": daily.Variables(7).ValuesAsNumpy(),
        "sunshine_duration": daily.Variables(8).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(9).ValuesAsNumpy(),
        "rain_sum": daily.Variables(10).ValuesAsNumpy(),
        "snowfall_sum": daily.Variables(11).ValuesAsNumpy(),
        "precipitation_hours": daily.Variables(12).ValuesAsNumpy(),
        "wind_speed_10m_max": daily.Variables(13).ValuesAsNumpy(),
        "wind_gusts_10m_max": daily.Variables(14).ValuesAsNumpy(),
        "wind_direction_10m_dominant": daily.Variables(15).ValuesAsNumpy(),
        "shortwave_radiation_sum": daily.Variables(16).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": daily.Variables(17).ValuesAsNumpy()
    })
    return daily_data

def process_hourly_weather_data(response):
    hourly_response = response[0]
    hourly = hourly_response.Hourly()
    hourly_data = pd.DataFrame({
        "datetime": pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True), end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "dew_point_2m": hourly.Variables(2).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(3).ValuesAsNumpy(),
        "precipitation": hourly.Variables(4).ValuesAsNumpy(),
        "rain": hourly.Variables(5).ValuesAsNumpy(),
        "snowfall": hourly.Variables(6).ValuesAsNumpy(),
        "snow_depth": hourly.Variables(7).ValuesAsNumpy(),
        "weather_code": hourly.Variables(8).ValuesAsNumpy(),
        "pressure_msl": hourly.Variables(9).ValuesAsNumpy(),
        "surface_pressure": hourly.Variables(10).ValuesAsNumpy(),
        "cloud_cover": hourly.Variables(11).ValuesAsNumpy(),
        "cloud_cover_low": hourly.Variables(12).ValuesAsNumpy(),
        "cloud_cover_mid": hourly.Variables(13).ValuesAsNumpy(),
        "cloud_cover_high": hourly.Variables(14).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": hourly.Variables(15).ValuesAsNumpy(),
        "vapour_pressure_deficit": hourly.Variables(16).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(17).ValuesAsNumpy(),
        "wind_speed_100m": hourly.Variables(18).ValuesAsNumpy(),
        "wind_direction_10m": hourly.Variables(19).ValuesAsNumpy(),
        "wind_direction_100m": hourly.Variables(20).ValuesAsNumpy(),
        "wind_gusts_10m": hourly.Variables(21).ValuesAsNumpy(),
        "soil_temperature_0_to_7cm": hourly.Variables(22).ValuesAsNumpy(),
        "soil_temperature_7_to_28cm": hourly.Variables(23).ValuesAsNumpy(),
        "soil_temperature_28_to_100cm": hourly.Variables(24).ValuesAsNumpy(),
        "soil_temperature_100_to_255cm": hourly.Variables(25).ValuesAsNumpy(),
        "soil_moisture_0_to_7cm": hourly.Variables(26).ValuesAsNumpy(),
        "soil_moisture_7_to_28cm": hourly.Variables(27).ValuesAsNumpy(),
        "soil_moisture_28_to_100cm": hourly.Variables(28).ValuesAsNumpy(),
        "soil_moisture_100_to_255cm": hourly.Variables(29).ValuesAsNumpy()
    })
    return hourly_data

def save_data_to_csv(dataframe, filename):
    dataframe.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")
