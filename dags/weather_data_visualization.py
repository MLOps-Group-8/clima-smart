import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from google.cloud import storage
import io

from util import read_csv_from_gcs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_plots(bucket_name, daily_blob_name, hourly_blob_name, output_dir):
    """
    Generates and saves various weather data plots.

    Args:
    bucket_name (str): Name of the GCS bucket.
    daily_blob_name (str): Blob name for the daily data CSV.
    output_dir (str): Directory to save the plots.

    Uses daily weather data to generate a plot of maximum daily temperatures, 
    hourly temperature changes, and a histogram of wind speeds.
    """
    logging.info("Generating visualizations from weather data.")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    daily_data = read_csv_from_gcs(bucket_name, daily_blob_name)
    hourly_data = read_csv_from_gcs(bucket_name, hourly_blob_name)
    
    
    # Convert 'date' and 'datetime' from string to datetime objects
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    hourly_data['datetime'] = pd.to_datetime(hourly_data['datetime'])
    
    # Plot for Daily Maximum Temperatures
    buf = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_data['date'], daily_data['temperature_2m_max'], label='Daily Max Temp', color='red')
    plt.title('Daily Maximum Temperatures Over Time')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (°F)')
    plt.legend()
    plt.grid(True)
    plt.savefig(buf, format='png')
    buf.seek(0)  # Rewind the buffer

    blob = bucket.blob(f'{output_dir}/daily_max_temperatures.png')
    blob.upload_from_file(buf, content_type='image/png')
    plt.close()
    buf.close()
    logging.info('Daily maximum temperature plot uploaded to GCS.')

    # Plot for Hourly Temperature Changes
    specific_day = hourly_data['datetime'].dt.date == hourly_data['datetime'].dt.date.min()
    buf = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_data[specific_day]['datetime'], hourly_data[specific_day]['temperature_2m'], label='Hourly Temp', color='blue')
    plt.title('Hourly Temperature Changes on First Available Day')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°F)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig(buf, format='png')
    buf.seek(0)  # Rewind the buffer

    blob = bucket.blob(f'{output_dir}/hourly_temperature_changes.png')
    blob.upload_from_file(buf, content_type='image/png')
    plt.close()
    buf.close()
    logging.info('Hourly temperature change plot uploaded to GCS.')

    # 3. Histogram of wind speeds
    buf = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.hist(hourly_data['wind_speed_10m'], bins=30, alpha=0.7)
    plt.title('Histogram of Wind Speeds')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(buf, format='png')
    buf.seek(0)  # Rewind the buffer

    blob = bucket.blob(f'{output_dir}/wind_speed_histogram.png')
    blob.upload_from_file(buf, content_type='image/png')
    plt.close()
    buf.close()
    logging.info('Histogram of wind speeds uploaded to GCS.')


    logging.info(f"Visualizations saved to {output_dir}.")

    
def get_season(month):
    """
    Determines the meteorological season for a given month.

    Args:
    month (int): The month as an integer.

    Returns:
    str: The meteorological season name.
    """
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

def generate_seasonal_trend_plots(bucket_name, daily_blob_name, output_dir):
    logging.info(f"Generating seasonal trend plots for data in {daily_blob_name}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    # Load data from GCS
    daily_data = read_csv_from_gcs(bucket_name, daily_blob_name)
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    daily_data['season'] = daily_data['date'].dt.month.apply(get_season)

    # Aggregate data by season and year
    seasonal_data = daily_data.groupby([daily_data['date'].dt.year, 'season']).agg({
        'temperature_2m_max': 'mean',
        'temperature_2m_min': 'mean',
        'precipitation_sum': 'sum'
    }).reset_index()

    # Rename columns for clarity
    seasonal_data.rename(columns={
        'date': 'year',
        'temperature_2m_max': 'avg_max_temp',
        'temperature_2m_min': 'avg_min_temp',
        'precipitation_sum': 'total_precip'
    }, inplace=True)

    # Initialize Google Cloud Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Generate and upload plots
    metrics = [('avg_max_temp', 'Temperature (°F)'), ('avg_min_temp', 'Temperature (°F)'), ('total_precip', 'Precipitation (mm)')]
    for metric, ylabel in metrics:
        for plot_type in ['max', 'min', 'total']:
            buf = io.BytesIO()
            plt.figure(figsize=(14, 6))
            for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
                subset = seasonal_data[seasonal_data['season'] == season]
                plt.plot(subset['year'], subset[metric], marker='o', label=f'{season} {plot_type.capitalize()}')
            plt.title(f'Seasonal {plot_type.capitalize()} {ylabel}')
            plt.xlabel('Year')
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)  # Rewind the buffer

            # Construct blob path
            plot_filename = f'{plot_type}_{metric}.png'
            blob = bucket.blob(f'{output_dir}/{plot_filename}')
            blob.upload_from_file(buf, content_type='image/png')
            plt.close()
            buf.close()
            logging.info(f'Plot {plot_filename} uploaded to GCS at {output_dir}/{plot_filename}.')

