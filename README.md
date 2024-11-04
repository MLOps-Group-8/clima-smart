# ClimaSmart


ClimaSmart is a next-generation AI-powered weather prediction system designed to provide highly accurate, real-time weather forecasts alongside personalized recommendations. The system offers features such as Weather-Based Commute Suggestions, Outdoor Adventure Recommendations, and an interactive chatbot for real-time queries about weather conditions.

## Workflow
![WorkFlow Diagram](reports/figures/Worflow.png)

## Project Scope
[Project Scope Document](reports/Project_Scoping_Group_8.pdf)

## Project Organization

```
├── LICENSE                <- Open-source license if one is chosen
├── Makefile               <- Makefile with convenience commands like `make data` or `make train`
├── README.md              <- The top-level README for developers using this project.
├── data
│   ├── external           <- Data from third party sources.
│   ├── interim            <- Intermediate data that has been transformed.
│   ├── processed          <- The final, canonical data sets for modeling.
│   └── raw                <- The original, immutable data dump.
│
├── docs                   <- A default mkdocs project; see www.mkdocs.org for details
│
├── dags                   <- Directory for Airflow DAGs and utility scripts
│   ├── util.py            <- Utility functions used across different scripts
│   ├── weather_data_collection.py           <- Script to collect weather data
│   ├── weather_data_collection_dag.py       <- Airflow DAG for scheduling data collection
│   ├── weather_data_preprocessing.py        <- Script for data preprocessing
│   └── weather_data_visualization.py        <- Script for generating data visualizations
│
├── models                 <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks              <- Jupyter notebooks for exploratory data analysis and initial prototyping
│   ├── 1.0-jqp-initial-data-exploration.ipynb
│
├── pyproject.toml         <- Project configuration file with package metadata for clima_smart and configuration for tools like black
│
├── references             <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
│
├── setup.cfg              <- Configuration file for flake8
│
└── clima_smart            <- Source code for use in this project.
    ├── __init__.py        <- Makes clima_smart a Python module
    ├── config.py          <- Store useful variables and configuration
    ├── dataset.py         <- Scripts to download or generate data
    ├── features.py        <- Code to create features for modeling
    ├── modeling           <- Subpackage for model training and prediction
    │   ├── __init__.py
    │   ├── predict.py     <- Code to run model inference with trained models          
    │   └── train.py       <- Code to train models
    └── plots.py           <- Code to create visualizations
```

# Weather Data Collection DAG

This document describes the Airflow DAG `weather_data_collection_dag.py`, which automates the process of collecting, processing, and storing weather data. The DAG ensures data is fetched regularly and reliably, leveraging Apache Airflow's scheduling and monitoring capabilities.

## Overview

The DAG defined in this module handles the orchestration of tasks related to:
- Fetching raw weather data from external APIs. [https://open-meteo.com/en/docs/historical-weather-api]
- Processing the raw data into a structured format.
- Optionally, uploading processed data to a cloud storage solution or a database.

## Dependencies

This DAG relies on the following Python modules and Airflow operators:

- `pandas`: For data manipulation.
- `requests` or `http_lib`: Depending on the method of HTTP requests to APIs.
- Airflow operators such as `PythonOperator`, `BashOperator`, etc.

Make sure to install the necessary packages:

```bash
pip install apache-airflow pandas requests
```

# Weather Data Collection Module

This module, `weather_data_collection.py`, is part of a larger project aimed at collecting, processing, and visualizing weather data. The script automates the retrieval of weather data from various APIs and processes it for further analysis and visualization.

## Overview

The module performs the following key functions:

- **API Data Fetching**: Connects to weather data sources and fetches data based on specified parameters.
- **Data Processing**: Cleans and transforms raw data into a usable format for analysis.
- **Data Storage**: Stores the processed data in a structured format, ready for analysis or visualization.

## Dependencies

This module requires the following Python libraries:

- `pandas`: For data manipulation and analysis.
- `requests`: For making HTTP requests to weather APIs.
- `matplotlib`: (Optional) For generating any initial exploratory plots.

Ensure these dependencies are installed using:

```bash
pip install pandas requests matplotlib
```

# Weather Data Preprocessing Module

This module, `weather_data_preprocessing.py`, focuses on cleaning, transforming, and preparing weather data for further analysis or visualization. It is designed to ensure data quality and usability by addressing common issues such as missing values, incorrect formats, and data inconsistencies.

## Overview

The preprocessing module is crucial for:
- Ensuring data accuracy and consistency.
- Transforming raw data into a format suitable for analysis.
- Enhancing data quality through cleaning operations.

## Dependencies

This script depends on several Python libraries:

- `pandas`: For efficient data manipulation and analysis.
- `numpy`: For numerical operations.
- Additional libraries might include `scikit-learn` for scaling and normalization if used.

Ensure these are installed using:

```bash
pip install pandas numpy scikit-learn
```

# Weather Data Visualization Module

This module, `weather_data_visualization.py`, is designed to create visual representations of weather data to aid in analysis and reporting. It focuses on generating various types of plots that highlight trends, patterns, and anomalies in the weather data collected and processed by other components of the project.

## Overview

The visualization module serves to:
- Provide insights through visual means which are easily digestible.
- Support exploratory data analysis with graphical representations.
- Enable stakeholders to make informed decisions based on visual trends.

## Dependencies

This script relies on several libraries, prominently:

- `matplotlib`: For creating static, animated, and interactive visualizations.
- `seaborn`: For high-level interface for drawing attractive and informative statistical graphics.
- `pandas`: For data manipulation necessary before visualization.

Make sure these are installed using:

```bash
pip install matplotlib seaborn pandas
```

--------

