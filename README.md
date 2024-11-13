# ClimaSmart

ClimaSmart is a next-generation AI-powered weather prediction system designed to provide highly accurate, real-time weather forecasts alongside personalized recommendations. The system offers features such as Weather-Based Commute Suggestions, Outdoor Adventure Recommendations, and an interactive chatbot for real-time queries about weather conditions.

## Data Card and Dataset Information

### Dataset Source
Our dataset source is the [OpenMeteo API](https://archive-api.open-meteo.com/v1/archive), which provides comprehensive historical weather data. We have utilized both hourly and daily weather data, covering data from the present date back to the year 2000. The data is used to capture a variety of weather conditions at different resolutions for robust prediction and analysis.

### Data Card - Daily
| Variable Name             | Role     | Type   | Description                                     | Units | Missing Values |
|---------------------------|----------|--------|-------------------------------------------------|-------|----------------|
| date                      | Feature  | Date   | Date of observation                             | -     | No             |
| temperature_2m_max        | Feature  | Float  | Daily maximum temperature                       | °C    | No             |
| temperature_2m_min        | Feature  | Float  | Daily minimum temperature                       | °C    | No             |
| daylight_duration         | Feature  | Float  | Duration of daylight                            | Hours | Yes            |
| sunshine_duration         | Feature  | Float  | Duration of sunshine                            | Hours | Yes            |
| precipitation_sum         | Feature  | Float  | Total precipitation                             | mm    | No             |
| rain_sum                  | Feature  | Float  | Total rainfall                                  | mm    | No             |
| showers_sum               | Feature  | Float  | Total showers amount                            | mm    | Yes            |
| sunshine_ratio            | Feature  | Float  | Ratio of sunshine hours to daylight hours       | -     | Yes            |
| is_weekend                | Feature  | Binary | Whether the date falls on a weekend (0 = No, 1 = Yes) | -     | No             |
| is_hot_day                | Feature  | Binary | Whether it was a hot day (0 = No, 1 = Yes)      | -     | No             |
| is_cold_day               | Feature  | Binary | Whether it was a cold day (0 = No, 1 = Yes)     | -     | No             |
| is_rainy_day              | Feature  | Binary | Whether it was a rainy day (0 = No, 1 = Yes)    | -     | No             |
| is_heavy_precipitation    | Feature  | Binary | Whether there was heavy precipitation (0 = No, 1 = Yes) | -     | No             |

### Data Card - Hourly
| Variable Name             | Role     | Type       | Description                                     | Units  | Missing Values |
|---------------------------|----------|------------|-------------------------------------------------|--------|----------------|
| datetime                  | Feature  | DateTime   | Date and time of observation                    | -      | No             |
| temperature_2m            | Feature  | Float      | Hourly temperature at 2 meters                  | °C     | No             |
| relative_humidity_2m      | Feature  | Float      | Hourly relative humidity at 2 meters            | %      | No             |
| dew_point_2m              | Feature  | Float      | Hourly dew point temperature at 2 meters        | °C     | No             |
| precipitation             | Feature  | Float      | Hourly precipitation                            | mm     | No             |
| rain                      | Feature  | Float      | Hourly rainfall                                 | mm     | No             |
| snowfall                  | Feature  | Float      | Hourly snowfall                                 | mm     | No             |
| cloud_cover               | Feature  | Float      | Cloud cover percentage                          | %      | No             |
| wind_speed_10m            | Feature  | Float      | Wind speed at 10 meters                         | m/s    | No             |
| wind_direction_10m        | Feature  | Float      | Wind direction at 10 meters                     | Degrees| No             |
| is_day                    | Feature  | Binary     | Whether it is day (0 = No, 1 = Yes)             | -      | No             |
| is_freezing               | Feature  | Binary     | Whether it is freezing (0 = No, 1 = Yes)        | -      | No             |
| is_raining                | Feature  | Binary     | Whether it is raining (0 = No, 1 = Yes)         | -      | No             |
| is_snowing                | Feature  | Binary     | Whether it is snowing (0 = No, 1 = Yes)         | -      | No             |
| temp_rolling_mean_24h     | Feature  | Float      | 24-hour rolling mean temperature                | °C     | Yes            |
| precip_rolling_sum_24h    | Feature  | Float      | 24-hour rolling sum of precipitation            | mm     | Yes            |
| wind_category             | Feature  | Categorical| Wind category based on speed                    | -      | Yes            |

## Project Scope
[Project Scope Document](reports/Project_Scoping_Group_8.pdf)

## Git Repository Structure

.
├── LICENSE                         ## License information
├── README.md                       ## Overview of the project, how to use it, and dataset details
├── assets                          ## Store images, graphs, or other visualizations
│   ├── images                      ## Images used in the project
│   └── plots                       ## Graphs or plots generated during analysis
├── config                          ## Configuration files for the project
│   ├── config.yaml                 ## General configurations like file paths, hyperparameters
│   ├── constants.py                ## Constant variables (e.g., target variable, feature names)
│   ├── Key.json                    ## Key file (possibly for authentication or API access)
│   └── schema.yaml                 ## Dataset schema (columns and types)
├── data                            ## Raw and processed data files
│   ├── raw                         ## Raw dataset files
│   │   ├── bank                    ## Raw bank marketing data
│   │   └── bank-additional          ## Additional raw bank marketing data
│   ├── processed                   ## Processed data, train/test splits
│       ├── eda_plots               ## Exploratory Data Analysis plots
│       ├── correlation_matrix.csv  ## Correlation matrix in CSV format
│       ├── correlation_matrix.pkl  ## Correlation matrix in pickle format
│       ├── dataframe_description.csv  ## Description of the dataframe
│       ├── dataframe_info.csv      ## Information about the dataframe
│       ├── datatype_format_processed.csv  ## Datatype format after processing (CSV)
│       ├── datatype_format_processed.pkl  ## Datatype format after processing (Pickle)
│       ├── datatype_info_after.csv  ## Datatype information after processing (CSV)
│       ├── datatype_info_before.csv  ## Datatype information before processing (CSV)
│       ├── encoded_data.csv        ## Encoded data in CSV format
│       ├── encoded_data.pkl        ## Encoded data in pickle format
│       ├── outlier_handled_data.csv  ## Data after handling outliers (CSV)
│       ├── outlier_handled_data.pkl  ## Data after handling outliers (Pickle)
│       ├── processed_data.pkl      ## Final processed data in pickle format
│       ├── raw_data.csv            ## Raw dataset in CSV format
│       └── raw_data.pkl            ## Raw dataset in pickle format  
├── dags
│   ├── logs                        # Directory to store log files generated by Airflow during DAG execution.
│   │   ├── dag_id=DataPipeline         ## Logs related to DataPipeline DAG execution 
│   │   │   ├── dag_processor_manager       ## Logs related to DAG processor manager 
│   │   │   ├── download_data.log           ## Log for downloading data 
│   │   │   ├── eda.log                     ## Log for Exploratory Data Analysis 
│   │   │   └── outlier_handling.log        ## Log for outlier handling process 
│   │
│   ├── src                         # Main source directory containing all Python scripts for data preprocessing and │validation.
│   │   ├── __pycache__             # Directory for Python cache files automatically generated after running scripts.
│   │   │   └── # These files are generated by Python to speed up execution. No need to modify them manually.
│   │   │
│   │   ├── data_preprocessing      # Contains scripts for various data preprocessing tasks.
│   │   │   ├── __pycache__         # Cache files related to the data preprocessing module.
│   │   │   ├── __init__.py          # Initialization file for the `data_preprocessing` module.
│   │   │   ├── correlation_analysis.py  # Script to perform correlation analysis on features in the dataset.
│   │   │   │                           # Helps identify multicollinearity or redundant features.
│   │   │   ├── datatype_format.py       # Script to ensure correct data types for all columns in the dataset.
│   │   │   │                           # Handles type conversion and ensures consistency across datasets.
│   │   │   ├── encoding.py              # Script responsible for encoding categorical variables into numerical values.
│   │   │   │                           # Uses techniques like one-hot encoding or label encoding.
│   │   │   ├── outlier_handling.py      # Script to detect and handle outliers using statistical methods like z-score │or IQR.
│   │   │   ├── preprocessing_main.py    # Main script coordinating all preprocessing steps, such as handling missing │values,
│   │   │                                # encoding, and normalization of features.
│   │   │   ├── removing_duplicates.py   # Script to remove duplicate entries from the dataset to ensure data integrity.
│   │   │   ├── smote.py                 # Implements SMOTE (Synthetic Minority Over-sampling Technique) to address │class imbalance.
│   │
│   │   ├── Data_validation          # Contains scripts for validating the quality of input data before further │processing.
│   │       ├── __pycache__         # Cache files related to the data validation module.
│   │       ├── anomaly_detection.py     # Script to detect anomalies in the dataset using predefined rules or │statistical methods.
│   │       ├── data_schema_statistics_generation.py  # Script that generates statistics about dataset schema (e.g., │column types, missing values).
│   │       │                                        # Ensures incoming datasets adhere to expected formats.
│   │       ├── util_bank_marketing.py  # Utility functions specific to a bank marketing dataset, including custom │transformations or validations.
│   ├── airflow.py                   # Main Airflow DAG definition file that orchestrates different tasks such as │downloading data,
│   │                                # preprocessing it, validating it, and loading it into a target system. Defines │task dependencies,schedules, and triggers within Airflow.
│   ├── DownloadData.py              # Script responsible for downloading raw datasets from external sources like APIs │or cloud storage systems.Ensures datasets are available locally before starting any preprocessing tasks.
│   ├── eda.py                       # Script dedicated to performing Exploratory Data Analysis (EDA) on raw datasets. │Generates summary statistics,
│   │                                # visualizations, and other insights needed before applying transformations.
│   ├── HandlingNullValues.py        # Script specifically designed to handle missing values in datasets. Applies │techniques like mean/mode imputation,
│   │                                 # forward/backward filling, or other imputation methods based on business logic.
│   ├── LoadData.py                  # After preprocessing and validation steps are complete, this script loads cleaned │datasets into a target system,such as a database or cloud storage service. The cleaned data can then be used for model │training or reporting.
│   │
│   ├── util_bank_marketing.py        # Utility script containing helper functions specific to handling bank marketing │datasets. These functions may include custom transformations, feature engineering, or domain-specific validations.
│   └── __init__.py                  # Initialization file for the overall `dags` module. This file allows other modules │within `dags` to be imported easily.
├── plugins                         ## Plugins used in the project (if any)
├── src                             ## Source code for the project
│   ├── models                      ## Machine learning models and related scripts 
│   │   └── __init__.py             ## Initialization file for models module 
│   └── utils                       ## Utility functions used throughout the project 
│       └── __init__.py             ## Initialization file for utils module 
├── tests                           ## Unit tests for various modules 
│   ├── test_CorrelationAndEncoding.py  ## Test script for correlation and encoding 
│   ├── test_data_format.py         ## Test script for data formatting 
│   ├── test_DownloadAndLoadData.py  ## Test script for downloading and loading data 
│   ├── test_eda.py                 ## Test script for exploratory data analysis 
│   ├── test_HandleNullValues.py    ## Test script for handling null values 
│   └── test_Smote.py               ## Test script for SMOTE oversampling technique 
├── .dvcignore                      ## DVC ignore file to specify which files to exclude from DVC tracking 
├── .env                            ## Environment variables file 
├── .gitattributes                  ## Git attributes configuration file 
├── .gitignore                      ## Git ignore file to exclude specific files from version control 
├── airflow.cfg                     ## Airflow configuration file  
├── docker-compose.yaml             ## Docker Compose setup for multi-container environments 
├── Dockerfile                      ## Docker container setup  
└── requirements.txt                ## List of dependencies (Python libraries)  


## Installation

This project requires **Python 3.8 or higher**. Please ensure that the correct version of Python is installed on your device. This project is compatible with Windows, Linux, and macOS.

### Prerequisites

- **Git**
- **Python** >= 3.8
- **Docker Daemon/Desktop** (Docker must be running)

### User Installation

Follow these steps to install and set up the project:

1. **Clone the Git Repository**:
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/MLOps-Group-8/clima-smart.git 
   ```

2. **Verify Python Version**:
   Check if Python 3.8 or higher is installed:
   ```bash
   python --version
   ```

3. **Check Available Memory**:
   Run the following command to check if you have sufficient memory for Docker:
   ```bash
   docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
   ```
   > **Note**: If you encounter an error like "Task exited with return code -9 or zombie job," you may need to increase the memory allocation for Docker.

4. **Edit the `docker-compose.yaml` File**:
   After cloning the repository, make the following changes in the `docker-compose.yaml` file as needed:
   - **User Permissions**:
     ```yaml
     user: "1000:0"
     ```
     This setting is usually sufficient, but if you encounter permission issues, modify it based on your `uid` and `gid`.
   - **SMTP Settings**:
     If you’re using a non-Gmail email provider for notifications, adjust these values accordingly:
     ```yaml
     AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com # Change if not using Gmail
     AIRFLOW__SMTP__SMTP_USER: # Your email address (no quotes)
     AIRFLOW__SMTP__SMTP_PASSWORD: # App password generated from your email provider
     AIRFLOW__SMTP__SMTP_MAIL_FROM: # Your email address
     ```
   - **Directory Mappings**:
     Set up the paths for your `dags`, `logs`, and `config` directories:
     ```yaml
     - ${AIRFLOW_PROJ_DIR:-.}/dags: # Full path to your DAGs folder
     - ${AIRFLOW_PROJ_DIR:-.}/logs: # Full path to your logs folder
     - ${AIRFLOW_PROJ_DIR:-.}/config: # Full path to the config folder
     ```

5. **Add GCP Credentials**:
   In the cloned project directory, navigate to the `config` folder and place your `key.json` file from the Google Cloud Platform (GCP) service account here. This will enable data access from GCP.

6. **Build the Docker Image**:
   Build the Docker image for the project:
   ```bash
   docker compose build
   ```

7. **Initialize Airflow**:
   Start Docker and initialize Airflow:
   ```bash
   docker compose up airflow-init
   ```

8. **Run the Docker Containers**:
   Start the Docker containers:
   ```bash
   docker compose up
   ```

9. **Access the Airflow Web Server**:
   Visit [http://localhost:8080](http://localhost:8080) in your web browser to access the Airflow UI. Use the following credentials to log in:
   - **Username**: `airflow2`
   - **Password**: `airflow2`

   Once logged in, start the DAG by clicking the play button next to the DAG name.

10. **Stop Docker Containers**:
    To stop the running Docker containers, press `Ctrl + C` in the terminal.
---

# Data Pipeline for Weather Collection, Processing, and Analysis

This section provides details on the ClimaSmart Data Pipeline, which uses **Apache Airflow** within **Docker** for orchestration. The pipeline automates weather data collection, processing, feature engineering, and quality validation to support ClimaSmart’s AI-powered weather prediction. The pipeline is designed to send email notifications upon successful completion of tasks or in the event of any errors, enhancing monitoring and alerting capabilities.

## Platform Overview

- **Platform**: Local environment using Docker containers
- **Orchestration**: Apache Airflow in Docker
- **Pipeline Objective**: Automated weather data collection, processing, quality validation, and notification alerts for data-driven insights.

## Pipeline Architecture

The pipeline consists of six core steps, each handled by a separate Directed Acyclic Graph (DAG) in Airflow. Each DAG automates specific tasks to streamline weather data processing from initial collection to final validation. Notifications are configured to send email alerts upon success or failure for each DAG.

---

### DAG Workflow and Status

| DAG                              | Purpose                                      | Status               |
|----------------------------------|----------------------------------------------|----------------------|
| Fetch and Save Daily Weather     | Retrieves daily data                         | Successfully Running |
| Fetch and Save Hourly Weather    | Retrieves hourly data                        | Successfully Running |
| Preprocess Daily Data            | Handles missing values and transformations   | Successfully Running |
| Preprocess Hourly Data           | Handles missing values and transformations   | Successfully Running |
| Feature Engineering              | Creates new features for analysis            | Successfully Running |
| EDA and Visualization            | Generates visualizations for data insights   | Successfully Running |
| Generate and Save Schema Stats   | Generates schema metadata                    | Successfully Running |
| Validate Weather Data            | Ensures schema and data quality              | Successfully Running |
| Test Data Quality and Schema     | Validates data quality against standards     | Successfully Running |
| Send Email Notification          | Sends email notifications upon task completion or error | Successfully Running |

---

### Detailed Workflow of Each Step

#### 1. Fetch and Save Daily Weather Data
- **Objective**: Retrieve daily weather data from external APIs and store it in Google Cloud Storage (GCS).
- **Process**:
  - **Data Collection**: Uses APIs like Open-Meteo’s Historical Weather API with the `requests` library.
  - **Data Processing**: Structures data with **pandas**, applying time zone standardization and formatting.
  - **Storage**: Saves as CSV in GCS.
- **DAG Status**: Successfully running, scheduled to run daily.
- **Notifications**: Sends email alerts upon successful completion or failure.

#### 2. Fetch and Save Hourly Weather Data
- **Objective**: Capture hourly weather data for granular analysis.
- **Process**:
  - **Data Collection**: Fetches hourly data via API calls with `requests`.
  - **Data Processing**: Formats data with **pandas**, aggregating and aligning time zones.
  - **Storage**: Saves in CSV format in GCS.
- **DAG Status**: Successfully running, scheduled to run hourly.
- **Notifications**: Sends email alerts upon successful completion or failure.

#### 3. Preprocess Daily Data
- **Objective**: Perform data preprocessing for daily weather data to ensure quality.
- **Process**:
  - **Handling Missing Values**: Uses **pandas** and **numpy** to fill or drop null values.
  - **Data Transformation**: Standardizes temperature units and date formats.
- **DAG Status**: Successfully running.
- **Notifications**: Sends email alerts upon successful completion or failure.

#### 4. Preprocess Hourly Data
- **Objective**: Perform data preprocessing for hourly weather data to ensure quality.
- **Process**:
  - **Handling Missing Values**: Uses **pandas** and **numpy** to fill or drop null values.
  - **Data Transformation**: Standardizes temperature units and date formats.
- **DAG Status**: Successfully running.
- **Notifications**: Sends email alerts upon successful completion or failure.

#### 5. Feature Engineering
- **Objective**: Enrich the data with new features for enhanced analysis.
- **Process**:
  - **Feature Generation**: Generates rolling averages, temperature differentials, etc., using **pandas**.
- **DAG Status**: Successfully running.
- **Notifications**: Sends email alerts upon successful completion or failure.

#### 6. EDA and Visualization
- **Objective**: Create visualizations to gain insights into the data.
- **Process**:
  - **Visualization**: Generates time series plots, heatmaps, and histograms using **matplotlib** and **seaborn**.
- **DAG Status**: Successfully running.
- **Notifications**: Sends email alerts upon successful completion or failure.

#### 7. Generate and Save Schema Statistics
- **Objective**: Generate summary statistics and schema metadata for data validation.
- **Process**:
  - **Schema Statistics**: Calculates min, max, and mean for each feature.
  - **Storage**: Saves schema statistics in GCS for reference.
- **DAG Status**: Successfully running.
- **Notifications**: Sends email alerts upon successful completion or failure.

#### 8. Validate Weather Data
- **Objective**: Ensure that weather data meets schema requirements.
- **Process**:
  - **Schema Validation**: Ensures data conforms to expected column names and types.
- **DAG Status**: Successfully running.
- **Notifications**: Sends email alerts upon successful completion or failure.

#### 9. Test Data Quality and Schema
- **Objective**: Conduct additional data quality checks.
- **Process**:
  - **Quality Testing**: Tests for duplicates, null values, and schema compliance using **pandas**.
- **DAG Status**: Successfully running.
- **Notifications**: Sends email alerts upon successful completion or failure.

#### 10. Send Email Notification
- **Objective**: Notify users of task completion status (success or error).
- **Process**:
  - **Email Alerts**: Sends email notifications to designated recipients based on task completion status.
- **DAG Status**: Successfully running.

---

### Technology Stack and Dependencies

- **Orchestration**: Apache Airflow within Docker
- **Storage**: Google Cloud Storage (GCS) for data files and schema metadata.
- **Python Libraries**:
  - **pandas** and **numpy** for data handling.
  - **requests** for API interactions.
  - **matplotlib** and **seaborn** for visualization.
  - **scikit-learn** for outlier detection and normalization.

### Orchestration and Scheduling with Apache Airflow
Airflow DAGs manage each stage, enabling robust monitoring and modularity. Each task is triggered on a schedule:
- **Daily Pipeline**: Runs each morning for daily data.
- **Hourly Pipeline**: Runs hourly to capture real-time data.

### Email Notifications
Each DAG in the pipeline has email notifications configured to provide alerts on the completion status of tasks:
- **Success Alerts**: Sent upon successful completion of tasks.
- **Error Alerts**: Sent in case of any errors or failures in task execution.

---

## DVC Integration for Data Tracking and Version Control

Integrating **Data Version Control (DVC)** with ClimaSmart’s pipeline helps manage and track data versions. Key tracked files include raw data, processed data, feature-engineered datasets, and visualizations.

### DVC Setup
1. **Initialize DVC**:
   ```bash
   dvc init
   ```

2. **Add Remote Storage**:
   ```bash
   dvc remote add -d myremote gs://your-gcs-bucket/dvc-storage
   ```

3. **Track Data Files**:
   ```bash
   dvc add data/raw/daily_weather.csv
   dvc add data/raw/hourly_weather.csv
   dvc add data/processed/preprocessed_daily.csv
   dvc add data/processed/preprocessed_hourly.csv
   dvc add data/featured/engineered_data.csv
   ```

### DVC Benefits
Using DVC enhances reproducibility and traceability of data changes across the pipeline stages, with remote storage providing scalable data access.

---

## Conclusion

The ClimaSmart Data Pipeline offers a robust solution for automated weather data processing. Each Airflow DAG is independently managed, ensuring modularity, scalability, and reliability for data-driven applications. The addition of Docker-based orchestration and email notifications enhances flexibility and monitoring for local deployment.

For any further setup instructions or pipeline expansion, refer to [Detailed Pipeline Documentation](reports/Data_Pipeline_Group8.pdf).

---
