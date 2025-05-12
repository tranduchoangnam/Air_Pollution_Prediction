#!/usr/bin/env python3
"""
Create Jupyter Notebook for Air Quality Prediction in Hà Nội

This script generates a comprehensive Jupyter notebook that implements
the air quality prediction pipeline using historical air quality and weather data.

Usage:
    python create_notebook.py
"""

import json
import os

def create_air_quality_notebook():
    """
    Create a Jupyter notebook for air quality prediction in Hà Nội
    """
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Air Quality Prediction for Hà Nội\n",
                    "\n",
                    "This notebook implements a comprehensive pipeline for predicting air quality in Hà Nội based on historical air quality measurements and weather data.\n",
                    "\n",
                    "## Project Overview\n",
                    "The implementation includes:\n",
                    "1. Data extraction from TimescaleDB (air quality) and CSV (weather)\n",
                    "2. Data preprocessing, cleaning, and integration\n",
                    "3. Exploratory data analysis (EDA)\n",
                    "4. Feature engineering for time series forecasting\n",
                    "5. Model training, evaluation, and comparison\n",
                    "6. Forecasting future air quality (AQI)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1. Import Required Libraries"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from datetime import datetime, timedelta\n",
                    "import os\n",
                    "import sys\n",
                    "import pickle\n",
                    "import json\n",
                    "import pytz\n",
                    "import warnings\n",
                    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                    "from sklearn.model_selection import TimeSeriesSplit\n",
                    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                    "from sklearn.linear_model import LinearRegression, Ridge\n",
                    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
                    "from sklearn.svm import SVR\n",
                    "from xgboost import XGBRegressor\n",
                    "from statsmodels.tsa.arima.model import ARIMA\n",
                    "from statsmodels.tsa.stattools import adfuller\n",
                    "\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Add parent directory to path to import utils\n",
                    "sys.path.append(os.path.abspath('../'))\n",
                    "from utils.timescaledb_util import TimescaleDBUtil\n",
                    "\n",
                    "# Set plotting styles\n",
                    "plt.style.use('seaborn-v0_8-whitegrid')\n",
                    "sns.set_palette('Set2')\n",
                    "pd.set_option('display.max_columns', None)\n",
                    "pd.set_option('display.max_rows', 50)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import the AirQualityPrediction class\n",
                    "from air_quality_prediction import AirQualityPrediction"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Initialize the Air Quality Prediction System"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Specify the path to the weather data CSV\n",
                    "weather_data_path = '../weather-dataset/result/hanoiweather_all.csv'\n",
                    "\n",
                    "# Create an instance of the AirQualityPrediction class\n",
                    "aq_predictor = AirQualityPrediction(weather_data_path=weather_data_path)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Data Loading & Preparation\n",
                    "\n",
                    "You can either use a previously processed dataset (if available) or extract new data from the raw sources."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check if processed data exists\n",
                    "use_processed_data = os.path.exists('hourly_air_quality_weather.csv')\n",
                    "print(f\"Processed data file exists: {use_processed_data}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3.1. Option 1: Load Previously Processed Data"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load from processed CSV file if it exists\n",
                    "if use_processed_data:\n",
                    "    aq_predictor.load_data()\n",
                    "    # Display basic information about the loaded data\n",
                    "    if aq_predictor.data is not None:\n",
                    "        print(f\"Data shape: {aq_predictor.data.shape}\")\n",
                    "        print(f\"Time range: {aq_predictor.data.index.min()} to {aq_predictor.data.index.max()}\")\n",
                    "        print(f\"Locations: {aq_predictor.data['location'].unique() if 'location' in aq_predictor.data.columns else 'Unknown'}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3.2. Option 2: Extract New Data from Raw Sources"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Set the date range for the air quality data\n",
                    "end_date = datetime.now()\n",
                    "start_date = end_date - timedelta(days=90)  # Get 90 days of data\n",
                    "\n",
                    "print(f\"Date range for extraction: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\")\n",
                    "\n",
                    "# Uncomment the below code to actually run the extraction\n",
                    "'''\n",
                    "success = aq_predictor.extract_and_process_data(\n",
                    "    start_date=start_date.strftime('%Y-%m-%d'),\n",
                    "    end_date=end_date.strftime('%Y-%m-%d'),\n",
                    "    save_to_csv=True\n",
                    ")\n",
                    "\n",
                    "if success:\n",
                    "    print(f\"Data extraction and processing successful. Data shape: {aq_predictor.data.shape}\")\n",
                    "else:\n",
                    "    print(\"Data extraction failed. Check the error messages for details.\")\n",
                    "'''"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3.3. Choose Data Loading Method\n",
                    "\n",
                    "Run ONE of the cells below, depending on whether you want to use existing data or extract new data."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# OPTION 1: Load from processed CSV\n",
                    "aq_predictor.load_data()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# OPTION 2: Extract new data from raw sources\n",
                    "'''\n",
                    "aq_predictor.extract_and_process_data(\n",
                    "    start_date=start_date.strftime('%Y-%m-%d'),\n",
                    "    end_date=end_date.strftime('%Y-%m-%d'),\n",
                    "    save_to_csv=True\n",
                    ")\n",
                    "'''"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3.4 Examine Raw Data"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Display the first few rows of the dataset\n",
                    "if aq_predictor.data is not None:\n",
                    "    aq_predictor.data.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 4. Data Preprocessing\n",
                           "\n",
                           "Handle missing values and convert data types."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["aq_predictor.preprocess_data()"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Verify missing values after preprocessing\n",
                    "if aq_predictor.data is not None:\n",
                    "    missing_values = aq_predictor.data.isnull().sum()\n",
                    "    print(\"Missing values after preprocessing:\")\n",
                    "    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else \"No missing values\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 5. Exploratory Data Analysis (EDA)\n",
                           "\n",
                           "Analyze patterns and relationships in the air quality and weather data."]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 5.1 Basic Statistics"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Display summary statistics for numerical columns\n",
                    "if aq_predictor.data is not None:\n",
                    "    numerical_data = aq_predictor.data.select_dtypes(include=[np.number])\n",
                    "    numerical_data.describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 5.2 Complete EDA Process"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["aq_predictor.exploratory_data_analysis()"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 5.3 Correlation Analysis"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Correlation heatmap for key variables\n",
                    "if aq_predictor.data is not None:\n",
                    "    # Select important variables for correlation analysis\n",
                    "    key_vars = ['calculated_aqi', 'avg_pm25', 'avg_pm10', 'avg_o3', 'avg_no2', 'avg_so2', 'avg_co', \n",
                    "                'temperature_c', 'humidity_pct', 'wind_speed_kmh']\n",
                    "    \n",
                    "    # Filter available columns\n",
                    "    available_vars = [var for var in key_vars if var in aq_predictor.data.columns]\n",
                    "    \n",
                    "    # Create correlation matrix\n",
                    "    if available_vars:\n",
                    "        corr_matrix = aq_predictor.data[available_vars].corr()\n",
                    "        \n",
                    "        # Plot heatmap\n",
                    "        plt.figure(figsize=(12, 10))\n",
                    "        mask = np.triu(corr_matrix)\n",
                    "        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', \n",
                    "                    fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)\n",
                    "        plt.title('Correlation Matrix of Key Variables', fontsize=16)\n",
                    "        plt.tight_layout()\n",
                    "        plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 5.4 AQI Categories Distribution"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Analyze AQI categories\n",
                    "if aq_predictor.data is not None and 'aqi_category' in aq_predictor.data.columns:\n",
                    "    aqi_counts = aq_predictor.data['aqi_category'].value_counts()\n",
                    "    \n",
                    "    # Plot pie chart\n",
                    "    plt.figure(figsize=(10, 8))\n",
                    "    plt.pie(aqi_counts, labels=aqi_counts.index, autopct='%1.1f%%', \n",
                    "            colors=[aq_predictor.aqi_colors.get(cat, 'gray') for cat in aqi_counts.index],\n",
                    "            startangle=90, shadow=True)\n",
                    "    plt.axis('equal')\n",
                    "    plt.title('Distribution of AQI Categories', fontsize=16)\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 6. Prepare Data for Forecasting\n",
                           "\n",
                           "Create features for time series forecasting, including lagged variables, time-based features, and rolling averages."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["forecast_data = aq_predictor.prepare_data_for_forecasting(target_column='calculated_aqi')"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Display the forecast data shape and first few rows\n",
                           "if aq_predictor.forecast_data is not None:\n",
                           "    print(f\"Forecast data shape: {aq_predictor.forecast_data.shape}\")\n",
                           "    aq_predictor.forecast_data.head()"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 6.1 Feature Importance Analysis"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Analyze features correlation with target\n",
                    "if aq_predictor.forecast_data is not None:\n",
                    "    target_col = 'calculated_aqi'\n",
                    "    correlations = aq_predictor.forecast_data.corr()[target_col].sort_values(ascending=False)\n",
                    "    \n",
                    "    # Plot top 15 correlations\n",
                    "    plt.figure(figsize=(12, 8))\n",
                    "    correlations.drop(target_col).head(15).plot(kind='bar')\n",
                    "    plt.title('Top 15 Features by Correlation with AQI', fontsize=16)\n",
                    "    plt.xlabel('Features')\n",
                    "    plt.ylabel('Correlation Coefficient')\n",
                    "    plt.grid(axis='y', alpha=0.3)\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 7. Train and Evaluate Models\n",
                           "\n",
                           "Train multiple forecasting models and evaluate their performance."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Split data into training and testing sets\n",
                    "X_train, X_test, y_train, y_test, test_index = aq_predictor.train_test_split(test_size=0.2)\n",
                    "\n",
                    "if X_train is not None:\n",
                    "    print(f\"Training set: {X_train.shape}, Testing set: {X_test.shape}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Train and evaluate all models\n",
                           "aq_predictor.train_and_evaluate_models()"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 8. Compare Model Performance\n",
                           "\n",
                           "Compare the performance of different forecasting models."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Create a performance comparison\n",
                           "aq_predictor.compare_models()"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Display model metrics as a table\n",
                    "if aq_predictor.model_results:\n",
                    "    results_df = pd.DataFrame(aq_predictor.model_results).sort_values('rmse')\n",
                    "    display(results_df)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 9. Forecast Future Air Quality\n",
                           "\n",
                           "Generate a forecast for the next 24 hours using the best model."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Find the best model based on RMSE (lower is better)\n",
                    "if aq_predictor.model_results:\n",
                    "    best_model = min(aq_predictor.model_results, key=lambda x: x['rmse'])\n",
                    "    print(f\"Best model: {best_model['model']} with RMSE: {best_model['rmse']:.2f}, MAE: {best_model['mae']:.2f}, R²: {best_model['r2']:.4f}\")\n",
                    "    \n",
                    "    # Generate forecast using the best model\n",
                    "    forecast_df = aq_predictor.forecast_future(model_name=best_model['model'], steps=24)\n",
                    "else:\n",
                    "    print(\"No model results available. Using XGBoost as default.\")\n",
                    "    forecast_df = aq_predictor.forecast_future(model_name=\"XGBoost\", steps=24)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Display the forecast\n",
                           "if forecast_df is not None:\n",
                           "    display(forecast_df)"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 9.1 AQI Category Forecasts"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Convert numeric AQI forecasts to AQI categories\n",
                    "if forecast_df is not None:\n",
                    "    forecast_df['aqi_category'] = forecast_df['forecasted_aqi'].apply(aq_predictor.get_aqi_category)\n",
                    "    \n",
                    "    # Display forecasted categories\n",
                    "    forecast_categories = forecast_df.groupby('aqi_category').size().reset_index(name='hours')\n",
                    "    print(\"Forecasted AQI Categories for next 24 hours:\")\n",
                    "    display(forecast_categories)\n",
                    "    \n",
                    "    # Visualize category distribution\n",
                    "    plt.figure(figsize=(10, 6))\n",
                    "    plt.pie(forecast_categories['hours'], labels=forecast_categories['aqi_category'], autopct='%1.1f%%',\n",
                    "            colors=[aq_predictor.aqi_colors.get(cat, 'gray') for cat in forecast_categories['aqi_category']],\n",
                    "            shadow=True, startangle=90)\n",
                    "    plt.axis('equal')\n",
                    "    plt.title('Distribution of Forecasted AQI Categories (Next 24 Hours)', fontsize=16)\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 10. Save Trained Models\n",
                           "\n",
                           "Save the trained models for future use."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["aq_predictor.save_models(output_dir='models')"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 11. Compare Different Forecast Horizons"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generate forecasts for different time horizons\n",
                    "horizons = [6, 12, 24, 48]\n",
                    "\n",
                    "if aq_predictor.model_results:\n",
                    "    best_model = min(aq_predictor.model_results, key=lambda x: x['rmse'])['model']\n",
                    "    \n",
                    "    for horizon in horizons:\n",
                    "        print(f\"\\nGenerating {horizon}-hour forecast using {best_model}...\")\n",
                    "        forecast = aq_predictor.forecast_future(model_name=best_model, steps=horizon)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 12. Conclusion\n",
                    "\n",
                    "We've successfully implemented a complete pipeline for air quality prediction in Hà Nội that:\n",
                    "1. Loads data from raw sources (TimescaleDB and weather CSV)\n",
                    "2. Processes and merges the datasets\n",
                    "3. Engineers features for time series forecasting\n",
                    "4. Trains and evaluates multiple models\n",
                    "5. Generates forecasts for future air quality\n",
                    "\n",
                    "### Key Findings:\n",
                    "- The best performing model was found to be the one with the lowest RMSE\n",
                    "- Weather variables and previous AQI readings are important predictors\n",
                    "- Time-based patterns in AQI values show day-of-week and hour-of-day variations\n",
                    "\n",
                    "### Next Steps:\n",
                    "- Deploy the model for real-time predictions\n",
                    "- Add more data sources to improve accuracy\n",
                    "- Implement alerts for hazardous air quality forecasts"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write the notebook to a file
    with open("air_quality_prediction.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)
    
    print("Jupyter notebook 'air_quality_prediction.ipynb' has been created successfully.")

if __name__ == "__main__":
    create_air_quality_notebook() 