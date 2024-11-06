import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from google.cloud import storage
import logging
import traceback

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

def fetch_daily_weather_data(client, api_url, params):
    """
    Processes the raw API response for daily weather data into a structured pandas DataFrame.

    Args:
    response (list): The API response containing daily weather data.

    Returns:
    pandas.DataFrame: The processed daily weather data.
    """
    logging.info(f"Fetching daily weather data from {api_url} with params {params}")
    params.update({"daily": "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,daylight_duration,sunshine_duration,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration"})
    return client.weather_api(api_url, params=params)

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
    params.update({"hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,precipitation,rain,showers,snowfall,snow_depth,weather_code,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,visibility,evapotranspiration,et0_fao_evapotranspiration,vapour_pressure_deficit,wind_speed_10m,wind_speed_80m,wind_speed_120m,wind_speed_180m,wind_direction_10m,wind_direction_80m,wind_direction_120m,wind_direction_180m,wind_gusts_10m,temperature_80m,temperature_120m,temperature_180m,soil_temperature_0cm,soil_temperature_6cm,soil_temperature_18cm,soil_temperature_54cm,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm,soil_moisture_9_to_27cm,soil_moisture_27_to_81cm,uv_index,uv_index_clear_sky,is_day,sunshine_duration,cape,shortwave_radiation,shortwave_radiation_instant,direct_radiation_instant"})
    return client.weather_api(api_url, params=params)

def process_daily_weather_data(response):
    """
    Processes the raw API response for daily weather data into a structured pandas DataFrame.

    Args:
    response (list): The API response containing daily weather data.

    Returns:
    pandas.DataFrame: The processed daily weather data.
    """
    daily_response = response[0]
    daily = daily_response.Daily()
    daily_data = pd.DataFrame({
        "date": pd.date_range(start=pd.to_datetime(daily.Time(), unit="s", utc=True), end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=daily.Interval()), inclusive="left"),
        "weather_code": daily.Variables(0).ValuesAsNumpy() if daily.VariablesLength() > 0 else None,
        "temperature_2m_max": daily.Variables(1).ValuesAsNumpy() if daily.VariablesLength() > 1 else None,
        "temperature_2m_min": daily.Variables(2).ValuesAsNumpy() if daily.VariablesLength() > 2 else None,
        "apparent_temperature_max": daily.Variables(3).ValuesAsNumpy() if daily.VariablesLength() > 3 else None,
        "apparent_temperature_min": daily.Variables(4).ValuesAsNumpy() if daily.VariablesLength() > 4 else None,
        "sunrise": daily.Variables(5).ValuesAsNumpy() if daily.VariablesLength() > 5 else None,
        "sunset": daily.Variables(6).ValuesAsNumpy() if daily.VariablesLength() > 6 else None,
        "daylight_duration": daily.Variables(7).ValuesAsNumpy() if daily.VariablesLength() > 7 else None,
        "sunshine_duration": daily.Variables(8).ValuesAsNumpy() if daily.VariablesLength() > 8 else None,
        "uv_index_max": daily.Variables(9).ValuesAsNumpy() if daily.VariablesLength() > 9 else None,
        "uv_index_clear_sky_max": daily.Variables(10).ValuesAsNumpy() if daily.VariablesLength() > 10 else None,
        "precipitation_sum": daily.Variables(11).ValuesAsNumpy() if daily.VariablesLength() > 11 else None,
        "rain_sum": daily.Variables(12).ValuesAsNumpy() if daily.VariablesLength() > 12 else None,
        "showers_sum": daily.Variables(13).ValuesAsNumpy() if daily.VariablesLength() > 13 else None,
        "snowfall_sum": daily.Variables(14).ValuesAsNumpy() if daily.VariablesLength() > 14 else None,
        "precipitation_hours": daily.Variables(15).ValuesAsNumpy() if daily.VariablesLength() > 15 else None,
        "precipitation_probability_max": daily.Variables(16).ValuesAsNumpy() if daily.VariablesLength() > 16 else None,
        "wind_speed_10m_max": daily.Variables(17).ValuesAsNumpy() if daily.VariablesLength() > 17 else None,
        "wind_gusts_10m_max": daily.Variables(18).ValuesAsNumpy() if daily.VariablesLength() > 18 else None,
        "wind_direction_10m_dominant": daily.Variables(19).ValuesAsNumpy() if daily.VariablesLength() > 19 else None,
        "shortwave_radiation_sum": daily.Variables(20).ValuesAsNumpy() if daily.VariablesLength() > 20 else None,
        "et0_fao_evapotranspiration": daily.Variables(21).ValuesAsNumpy() if daily.VariablesLength() > 21 else None
    })
    return daily_data

def process_hourly_weather_data(response):
    """
    Processes the raw API response for hourly weather data into a structured pandas DataFrame.

    Args:
    response (list): The API response containing hourly weather data.

    Returns:
    pandas.DataFrame: The processed hourly weather data.
    """
    hourly_response = response[0]
    hourly = hourly_response.Hourly()
    hourly_data = pd.DataFrame({
        "datetime": pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True), end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy() if hourly.VariablesLength() > 0 else None,
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy() if hourly.VariablesLength() > 1 else None,
        "dew_point_2m": hourly.Variables(2).ValuesAsNumpy() if hourly.VariablesLength() > 2 else None,
        "apparent_temperature": hourly.Variables(3).ValuesAsNumpy() if hourly.VariablesLength() > 3 else None,
        "precipitation_probability": hourly.Variables(4).ValuesAsNumpy() if hourly.VariablesLength() > 4 else None,
        "precipitation": hourly.Variables(5).ValuesAsNumpy() if hourly.VariablesLength() > 5 else None,
        "rain": hourly.Variables(6).ValuesAsNumpy() if hourly.VariablesLength() > 6 else None,
        "showers": hourly.Variables(7).ValuesAsNumpy() if hourly.VariablesLength() > 7 else None,
        "snowfall": hourly.Variables(8).ValuesAsNumpy() if hourly.VariablesLength() > 8 else None,
        "snow_depth": hourly.Variables(9).ValuesAsNumpy() if hourly.VariablesLength() > 9 else None,
        "weather_code": hourly.Variables(10).ValuesAsNumpy() if hourly.VariablesLength() > 10 else None,
        "pressure_msl": hourly.Variables(11).ValuesAsNumpy() if hourly.VariablesLength() > 11 else None,
        "surface_pressure": hourly.Variables(12).ValuesAsNumpy() if hourly.VariablesLength() > 12 else None,
        "cloud_cover": hourly.Variables(13).ValuesAsNumpy() if hourly.VariablesLength() > 13 else None,
        "cloud_cover_low": hourly.Variables(14).ValuesAsNumpy() if hourly.VariablesLength() > 14 else None,
        "cloud_cover_mid": hourly.Variables(15).ValuesAsNumpy() if hourly.VariablesLength() > 15 else None,
        "cloud_cover_high": hourly.Variables(16).ValuesAsNumpy() if hourly.VariablesLength() > 16 else None,
        "visibility": hourly.Variables(17).ValuesAsNumpy() if hourly.VariablesLength() > 17 else None,
        "evapotranspiration": hourly.Variables(18).ValuesAsNumpy() if hourly.VariablesLength() > 18 else None,
        "et0_fao_evapotranspiration": hourly.Variables(19).ValuesAsNumpy() if hourly.VariablesLength() > 19 else None,
        "vapour_pressure_deficit": hourly.Variables(20).ValuesAsNumpy() if hourly.VariablesLength() > 20 else None,
        "wind_speed_10m": hourly.Variables(21).ValuesAsNumpy() if hourly.VariablesLength() > 21 else None,
        "wind_speed_80m": hourly.Variables(22).ValuesAsNumpy() if hourly.VariablesLength() > 22 else None,
        "wind_speed_120m": hourly.Variables(23).ValuesAsNumpy() if hourly.VariablesLength() > 23 else None,
        "wind_speed_180m": hourly.Variables(24).ValuesAsNumpy() if hourly.VariablesLength() > 24 else None,
        "wind_direction_10m": hourly.Variables(25).ValuesAsNumpy() if hourly.VariablesLength() > 25 else None,
        "wind_direction_80m": hourly.Variables(26).ValuesAsNumpy() if hourly.VariablesLength() > 26 else None,
        "wind_direction_120m": hourly.Variables(27).ValuesAsNumpy() if hourly.VariablesLength() > 27 else None,
        "wind_direction_180m": hourly.Variables(28).ValuesAsNumpy() if hourly.VariablesLength() > 28 else None,
        "wind_gusts_10m": hourly.Variables(29).ValuesAsNumpy() if hourly.VariablesLength() > 29 else None,
        "temperature_80m": hourly.Variables(30).ValuesAsNumpy() if hourly.VariablesLength() > 30 else None,
        "temperature_120m": hourly.Variables(31).ValuesAsNumpy() if hourly.VariablesLength() > 31 else None,
        "temperature_180m": hourly.Variables(32).ValuesAsNumpy() if hourly.VariablesLength() > 32 else None,
        "soil_temperature_0cm": hourly.Variables(33).ValuesAsNumpy() if hourly.VariablesLength() > 33 else None,
        "soil_temperature_6cm": hourly.Variables(34).ValuesAsNumpy() if hourly.VariablesLength() > 34 else None,
        "soil_temperature_18cm": hourly.Variables(35).ValuesAsNumpy() if hourly.VariablesLength() > 35 else None,
        "soil_temperature_54cm": hourly.Variables(36).ValuesAsNumpy() if hourly.VariablesLength() > 36 else None,
        "soil_moisture_0_to_1cm": hourly.Variables(37).ValuesAsNumpy() if hourly.VariablesLength() > 37 else None,
        "soil_moisture_1_to_3cm": hourly.Variables(38).ValuesAsNumpy() if hourly.VariablesLength() > 38 else None,
        "soil_moisture_3_to_9cm": hourly.Variables(39).ValuesAsNumpy() if hourly.VariablesLength() > 39 else None,
        "soil_moisture_9_to_27cm": hourly.Variables(40).ValuesAsNumpy() if hourly.VariablesLength() > 40 else None,
        "soil_moisture_27_to_81cm": hourly.Variables(41).ValuesAsNumpy() if hourly.VariablesLength() > 41 else None,
        "uv_index": hourly.Variables(42).ValuesAsNumpy() if hourly.VariablesLength() > 42 else None,
        "uv_index_clear_sky": hourly.Variables(43).ValuesAsNumpy() if hourly.VariablesLength() > 43 else None,
        "is_day": hourly.Variables(44).ValuesAsNumpy() if hourly.VariablesLength() > 44 else None,
        "sunshine_duration": hourly.Variables(45).ValuesAsNumpy() if hourly.VariablesLength() > 45 else None,
        "cape": hourly.Variables(46).ValuesAsNumpy() if hourly.VariablesLength() > 46 else None,
        "shortwave_radiation": hourly.Variables(47).ValuesAsNumpy() if hourly.VariablesLength() > 47 else None,
        "shortwave_radiation_instant": hourly.Variables(48).ValuesAsNumpy() if hourly.VariablesLength() > 48 else None,
        "direct_radiation_instant": hourly.Variables(49).ValuesAsNumpy() if hourly.VariablesLength() > 49 else None
    })
    return hourly_data

def save_data_to_csv(dataframe, filename):
    dataframe.to_csv(filename, index=False)
    logging.info(f"Data saved to {filename}")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logging.info(f"File {source_file_name} uploaded to {destination_blob_name}.")
    except Exception as e:
        logging.error(f"Error uploading file to GCS: {e}")
