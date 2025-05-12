# Air Quality Prediction for Hanoi

This project aims to predict air quality in Hanoi based on historical air quality measurements and weather data. It includes a comprehensive data pipeline that extracts data from multiple sources, processes it, and trains forecasting models to predict future Air Quality Index (AQI) values.

## Overview

The implementation follows these key steps:

1. **Data Extraction**:
   - Air quality data from TimescaleDB (via HTTP API)
   - Weather data from CSV files

2. **Data Preprocessing**:
   - Converting units (temperature to Celsius, extracting numeric values from wind speed and humidity)
   - Normalizing weather categories
   - Setting all weather data location to "Hanoi" for consistency
   - Calculating AQI values based on EPA standards

3. **Data Integration**:
   - Aggregating weather data to hourly intervals
   - Merging air quality and weather data based on timestamps
   - Resampling and interpolating to ensure consistent time intervals
   - Handling missing values

4. **Feature Engineering**:
   - Creating time-based features with cyclical encoding (hour, day of week, month)
   - Adding lagged features (previous hours' values) at multiple time scales
   - Creating rolling statistics (mean, std, min, max) with various windows
   - Adding seasonal decomposition features (trend, seasonal, residual)
   - Identifying peak traffic hours and other air quality-specific patterns

5. **Model Development**:
   - Proper time series splitting to prevent data leakage
   - Time series cross-validation for reliable model evaluation
   - Implementing multiple forecasting approaches:
     - Machine Learning (XGBoost, Random Forest, Gradient Boosting)
     - Statistical (ARIMA, SARIMAX)
     - Specialized time series (Prophet, VAR)
   - Providing uncertainty estimates for forecasts
   - Comprehensive model evaluation with multiple metrics

## Project Structure

- `air_quality_prediction.py`: Main class for data processing and modeling
- `run_air_quality_prediction.py`: Simple script to run the complete pipeline
- `air_quality_cli.py`: Command-line interface with various options
- `main.py`: Data crawling script (DO NOT MODIFY)
- `models/`: Directory for saved models
- `hourly_air_quality_weather.csv`: Processed merged dataset (generated)

## Data Sources

### Air Quality Data
- Source: TimescaleDB
- Measurements: PM2.5, PM10, O3, NO2, SO2, CO
- Granularity: Hourly
- Stations: Multiple monitoring stations across Hanoi

### Weather Data
- Source: CSV file (`../weather-dataset/result/hanoiweather_all.csv`)
- Features: Temperature, humidity, wind speed, weather condition
- Timestamps: Irregular intervals

## AQI Calculation

Air Quality Index (AQI) is calculated based on EPA standards for each pollutant. The overall AQI is the maximum of individual pollutant AQIs. The AQI categories are:

- 0-50: Good
- 51-100: Moderate
- 101-150: Unhealthy for Sensitive Groups
- 151-200: Unhealthy
- 201-300: Very Unhealthy
- 301-500: Hazardous

## Key Features and Improvements

### Enhanced Feature Engineering
- Cyclical encoding of time features for better representation of periodic patterns
- Multiple lagged features covering different temporal scales (hours to weeks)
- Rolling statistics to capture local dynamics
- Traffic pattern features (rush hours, weekends, etc.)
- Seasonal decomposition to extract trend and cyclical components

### Advanced Forecasting Methods
- Prophet model for capturing multiple seasonality patterns
- Improved SARIMAX implementation with automatic parameter selection
- Uncertainty quantification for all forecasts
- Ensemble methods that combine multiple models

### Robust Evaluation
- Time series cross-validation to prevent data leakage
- Multiple metrics including RMSE, MAE, category accuracy, and directional accuracy
- AQI category confusion matrix to evaluate health-impact predictions
- Comprehensive visualizations for model comparison

## Usage

### Basic Usage

```bash
# Run with default settings (using processed CSV if available, or raw data if not)
python run_air_quality_prediction.py
```

### Advanced Usage with Command-line Options

```bash
# Force using raw data instead of processed CSV
python air_quality_cli.py --use-raw

# Specify custom date range
python air_quality_cli.py --start-date 2023-01-01 --end-date 2023-03-31

# Use a specific model and forecast horizon
python air_quality_cli.py --model xgboost --forecast-steps 48

# Skip EDA and save trained models
python air_quality_cli.py --skip-eda --save-models
```

### Command-line Options

- **Data Source Options**:
  - `--use-raw`: Use raw data sources instead of processed CSV
  - `--weather-path`: Path to weather data CSV (default: `../weather-dataset/result/hanoiweather_all.csv`)

- **Date Range Options**:
  - `--days`: Number of days of historical data to use (default: 90)
  - `--start-date`: Start date for data extraction (YYYY-MM-DD)
  - `--end-date`: End date for data extraction (YYYY-MM-DD)

- **Model Options**:
  - `--model`: Model to use for forecasting (choices: `linear`, `ridge`, `random_forest`, `gbr`, `xgboost`, `arima`, `prophet`, default: `prophet`)
  - `--forecast-steps`: Number of hours to forecast ahead (default: 24)

- **Output Options**:
  - `--save-models`: Save trained models
  - `--output-dir`: Directory to save models (default: `models`)
  - `--skip-eda`: Skip exploratory data analysis

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- statsmodels
- prophet
- pmdarima
- psycopg2 (for TimescaleDB connection)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/air-quality-prediction.git
cd air-quality-prediction

# Install dependencies
pip install -r requirements.txt
```

## TimescaleDB Connection

The connection to TimescaleDB is managed through the `TimescaleDBUtil` class. The connection details are loaded from:

1. Environment variable `DB_URL`
2. `.env` file in the parent directory
3. Hardcoded default value (not recommended for production)

The connection URL format is: `postgres://username:password@hostname:port/database?sslmode=require`

## Important Notes

- `main.py` is used for crawling data and should not be modified.
- Use `run_air_quality_prediction.py` for the simple pipeline or `air_quality_cli.py` for more control over the process.

## Future Improvements

- Deep learning models (LSTM, Transformer) for complex pattern recognition
- Integration with real-time weather forecasts for better predictive power
- Spatial interpolation for missing stations
- Incorporation of additional features (traffic, pollution sources)
- Web dashboard for visualization
- Real-time prediction API with alerts

## License

This project is licensed under the MIT License - see the LICENSE file for details. 