import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import sys
import pytz
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
sys.path.append(os.path.abspath('../'))
from utils.timescaledb_util import TimescaleDBUtil

class AirQualityPrediction:
    """
    A class for air quality prediction in Hanoi using time series forecasting methods.
    This implementation handles:
    1. Data loading and preprocessing from raw sources
    2. Feature engineering
    3. Model training and evaluation
    4. Forecasting future air quality
    """
    
    def __init__(self, weather_data_path='../weather-dataset/result/hanoiweather_all.csv'):
        """
        Initialize the air quality prediction system.
        
        Args:
            weather_data_path: Path to the weather data CSV
        """
        self.weather_data_path = weather_data_path
        self.data = None
        self.raw_weather_data = None
        self.raw_air_quality_data = None
        self.processed_weather_data = None
        self.hourly_weather_data = None
        self.forecast_data = None
        self.models = {}
        self.model_results = []
        self.scaler = None
        self.db_util = None
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('Set2')
        
        # AQI category definitions
        self.aqi_categories = {
            (0, 50): 'Good',
            (51, 100): 'Moderate',
            (101, 150): 'Unhealthy for Sensitive Groups',
            (151, 200): 'Unhealthy',
            (201, 300): 'Very Unhealthy',
            (301, 500): 'Hazardous'
        }
        
        # AQI category colors
        self.aqi_colors = {
            'Good': 'green',
            'Moderate': 'yellow',
            'Unhealthy for Sensitive Groups': 'orange',
            'Unhealthy': 'red',
            'Very Unhealthy': 'purple',
            'Hazardous': 'maroon'
        }
        
    def convert_ugm3_to_ppm(self, conc, pollutant):
        factors = {
            'co': 0.000872,      # to ppm
            'so2': 0.381,        # to ppb
            'no2': 0.531,        # to ppb
            'o3': 0.5094        # to ppb
        }
        if pd.isna(conc) or pollutant.lower() not in factors:
            return conc  # return original or NaN
        return conc * factors[pollutant.lower()]
    
    def calculate_aqi(self, concentration, pollutant):
        """Calculate AQI value from pollutant concentration
        
        Args:
            concentration: Pollutant concentration
            pollutant: One of 'pm25', 'pm10', 'o3', 'co', 'so2', 'no2'
            
        Returns:
            AQI value
        """
        # Define breakpoints for each pollutant
        # Format: [concentration breakpoints], [corresponding AQI breakpoints]
        breakpoints = {
            'pm25': {
                'conc': [0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4],
                'aqi': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'pm10': {
                'conc': [0, 54, 154, 254, 354, 424, 504, 604],
                'aqi': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'o3': {
                'conc': [0, 54, 70, 85, 105, 200, 404, 504],
                'aqi': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'co': {
                'conc': [0, 4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4],
                'aqi': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'so2': {
                'conc': [0, 35, 75, 185, 304, 604, 804, 1004],
                'aqi': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'no2': {
                'conc': [0, 53, 100, 360, 649, 1249, 1649, 2049],
                'aqi': [0, 50, 100, 150, 200, 300, 400, 500]
            }
        }
        
        # Handle missing or invalid data
        if pd.isna(concentration) or concentration < 0:
            return np.nan
        
        # Get breakpoints for the specified pollutant
        if pollutant.lower() not in breakpoints:
            raise ValueError(f"Unsupported pollutant: {pollutant}")
            
        pollutant = pollutant.lower()
        if pd.isna(concentration) or concentration < 0:
            return np.nan

        if pollutant not in breakpoints:
            raise ValueError(f"Unsupported pollutant: {pollutant}")

        # Convert if needed
        if pollutant in ['co', 'so2', 'no2', 'o3']:
            concentration = self.convert_ugm3_to_ppm(concentration, pollutant)

        conc_breaks = breakpoints[pollutant]['conc']
        aqi_breaks = breakpoints[pollutant]['aqi']

        if concentration > conc_breaks[-1]:
            return aqi_breaks[-1]

        for i in range(len(conc_breaks) - 1):
            if conc_breaks[i] <= concentration <= conc_breaks[i + 1]:
                c_low, c_high = conc_breaks[i], conc_breaks[i + 1]
                i_low, i_high = aqi_breaks[i], aqi_breaks[i + 1]
                if c_high == c_low:
                    return i_low
                aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
                return round(aqi)

        return np.nan

    def calculate_overall_aqi(self, pollutant_values):
        """Calculate overall AQI from multiple pollutant measurements
        
        Args:
            pollutant_values: Dictionary with pollutant concentrations
                (keys: 'pm25', 'pm10', 'o3', 'co', 'so2', 'no2')
                
        Returns:
            Overall AQI value (maximum of individual AQIs)
        """
        aqi_values = {}
        
        # Calculate AQI for each pollutant
        for pollutant, concentration in pollutant_values.items():
            if pd.notna(concentration):
                try:
                    aqi_values[pollutant] = self.calculate_aqi(concentration, pollutant)
                except Exception as e:
                    print(f"Error calculating AQI for {pollutant}: {e}")
        
        # Return maximum AQI value if any are available
        if aqi_values:
            return max(aqi_values.values())
        else:
            return np.nan
        
    def get_aqi_category(self, aqi_value):
        """Get the AQI category based on the AQI value
        
        Args:
            aqi_value: AQI value
            
        Returns:
            AQI category string
        """
        if pd.isna(aqi_value):
            return "Unknown"
        
        if aqi_value <= 50:
            return "Good"
        elif aqi_value <= 100:
            return "Moderate"
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi_value <= 200:
            return "Unhealthy"
        elif aqi_value <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def add_aqi_columns(self, df):
        """Add AQI calculations to a DataFrame with pollutant measurements
        
        Args:
            df: DataFrame with columns like 'avg_pm25', 'avg_pm10', etc.
            
        Returns:
            DataFrame with added AQI columns
        """
        result_df = df.copy()
        
        # Map from DataFrame column names to pollutant names
        pollutant_map = {
            'avg_pm25': 'pm25',
            'avg_pm10': 'pm10',
            'avg_o3': 'o3',
            'avg_co': 'co',
            'avg_so2': 'so2',
            'avg_no2': 'no2'
        }
        
        # Calculate individual AQI values for each pollutant
        for col, pollutant in pollutant_map.items():
            if col in result_df.columns:
                aqi_col = f'aqi_{pollutant}'
                result_df[aqi_col] = result_df[col].apply(lambda x: self.calculate_aqi(x, pollutant) if pd.notna(x) else np.nan)
        
        # Calculate overall AQI
        result_df['calculated_aqi'] = result_df.apply(lambda row: self.calculate_overall_aqi({
            'pm25': row['avg_pm25'] if 'avg_pm25' in row and pd.notna(row['avg_pm25']) else np.nan,
            'pm10': row['avg_pm10'] if 'avg_pm10' in row and pd.notna(row['avg_pm10']) else np.nan,
            'o3': row['avg_o3'] if 'avg_o3' in row and pd.notna(row['avg_o3']) else np.nan,
            'co': row['avg_co'] if 'avg_co' in row and pd.notna(row['avg_co']) else np.nan,
            'so2': row['avg_so2'] if 'avg_so2' in row and pd.notna(row['avg_so2']) else np.nan,
            'no2': row['avg_no2'] if 'avg_no2' in row and pd.notna(row['avg_no2']) else np.nan,
        }), axis=1)
        
        # Add AQI category
        result_df['aqi_category'] = result_df['calculated_aqi'].apply(self.get_aqi_category)
        
        return result_df
    
    def connect_to_timescaledb(self):
        """
        Connect to TimescaleDB to access air quality data.
        
        Returns:
            Boolean indicating if connection was successful
        """
        try:
            # Initialize TimescaleDB connection
            self.db_util = TimescaleDBUtil()
            
            # Test connection
            connection_successful = self.db_util.connect()
            print(f"Database connection {'successful' if connection_successful else 'failed'}")
            
            return connection_successful
        except Exception as e:
            print(f"Error connecting to TimescaleDB: {e}")
            return False
    
    def load_air_quality_data(self, start_date=None, end_date=None, location=None):
        """
        Load air quality data from TimescaleDB.
        
        Args:
            start_date: Start date for data extraction (defaults to 30 days ago)
            end_date: End date for data extraction (defaults to now)
            location: Filter by specific station (optional)
            
        Returns:
            Boolean indicating if data loading was successful
        """
        try:
            # Set default dates if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            # Connect to TimescaleDB if not already connected
            if self.db_util is None:
                if not self.connect_to_timescaledb():
                    return False
            
            # Construct WHERE clause with optional location filter
            where_clause = f"time >= '{start_date}' AND time <= '{end_date}'"
            if location:
                where_clause += f" AND station = '{location}' AND data_type = 'stat'"
            
            # Define the aggregations we want to perform - preserving station/location information
            aggregations = [
                "station",  # Keep the station as a dimension rather than aggregating
                'AVG("PM2.5") AS avg_pm25',
                'AVG("PM10") AS avg_pm10',
                'AVG("O3") AS avg_o3',
                'AVG("NO2") AS avg_no2',
                'AVG("SO2") AS avg_so2',
                'AVG("CO") AS avg_co'
            ]
            
            # Get hourly data using time_bucket, grouped by station
            self.raw_air_quality_data = self.db_util.get_time_bucket_data(
                table_name="air_quality_measurements",
                time_column="time",
                interval="1 hour",
                aggregations=aggregations,
                where_clause=where_clause,
                group_by="station",  # Group by station to preserve granularity
                order_by="time_bucket, station"
            )
            
            # Calculate AQI values if data was retrieved successfully
            if self.raw_air_quality_data is not None and not self.raw_air_quality_data.empty:
                self.raw_air_quality_data = self.add_aqi_columns(self.raw_air_quality_data)
                
                # Rename station to location for consistency with weather data
                self.raw_air_quality_data.rename(columns={'station': 'location'}, inplace=True)
                
                print(f"Successfully retrieved {len(self.raw_air_quality_data)} hourly air quality records")
                return True
            else:
                print("No air quality data retrieved from TimescaleDB")
                return False
        except Exception as e:
            print(f"Error loading air quality data: {e}")
            return False
    
    def load_weather_data(self):
        """
        Load raw weather data from CSV file.
        
        Returns:
            Boolean indicating if data loading was successful
        """
        try:
            self.raw_weather_data = pd.read_csv(self.weather_data_path)
            print(f"Successfully loaded weather data with {len(self.raw_weather_data)} records")
            return True
        except Exception as e:
            print(f"Error loading weather data: {e}")
            return False
    
    def preprocess_weather_data(self):
        """
        Preprocess the raw weather data, including:
        - Converting timestamps to datetime
        - Extracting numeric values from string format
        - Categorizing weather conditions
        - Setting a common location ('Hanoi') for all entries
        
        Returns:
            Boolean indicating if preprocessing was successful
        """
        if self.raw_weather_data is None:
            print("No weather data loaded. Call load_weather_data() first.")
            return False
        
        try:
            # Make a copy to avoid modifying the original
            self.processed_weather_data = self.raw_weather_data.copy()
            
            # Convert timestamp string to datetime
            self.processed_weather_data['timestamp'] = pd.to_datetime(self.processed_weather_data['timestamp'])
            
            # Extract numeric values from temperature, humidity, and wind_speed columns
            # Temperature: extract numeric value and convert to Celsius
            self.processed_weather_data['temperature_c'] = self.processed_weather_data['temperature'].str.extract(r'([-+]?\d*\.\d+|\d+)').astype(float)
            
            # Humidity: extract numeric percentage
            self.processed_weather_data['humidity_pct'] = self.processed_weather_data['humidity'].str.extract(r'(\d+)').astype(float)
            
            # Wind speed: extract numeric value in km/h
            self.processed_weather_data['wind_speed_kmh'] = self.processed_weather_data['wind_speed'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
            
            # Set location to 'Hanoi' for all records (for consistency with merging approach)
            self.processed_weather_data['location'] = 'Hanoi'
            
            # Add hour timestamp for aggregation
            self.processed_weather_data['timestamp_hour'] = self.processed_weather_data['timestamp'].dt.floor('H')
            
            # Categorize weather conditions based on weather_icon
            weather_categories = {
                'clear-day': 'clear_sky',
                'clear-night': 'clear_night',
                'partly-cloudy-day': 'partly_cloudy',
                'partly-cloudy-night': 'partly_cloudy',
                'cloudy': 'cloudy',
                'rain': 'rain',
                'light-rain': 'rain',
                'thunderstorm': 'storm',
                'snow': 'snow',
                'fog': 'fog'
            }
            
            # Map weather icons to categories, with 'other' as default
            self.processed_weather_data['weather_category'] = self.processed_weather_data['weather_icon'].map(
                lambda x: next((v for k, v in weather_categories.items() if k in str(x).lower()), 'other')
            )
            
            print("Weather data preprocessing complete.")
            return True
        except Exception as e:
            print(f"Error preprocessing weather data: {e}")
            return False
    
    def aggregate_weather_to_hourly(self):
        """
        Aggregate weather data to hourly intervals to match air quality data.
        
        Returns:
            Boolean indicating if aggregation was successful
        """
        if self.processed_weather_data is None:
            print("No processed weather data available. Call preprocess_weather_data() first.")
            return False
        
        try:
            # Group by hour, then aggregate (all data now has location="Hanoi")
            self.hourly_weather_data = self.processed_weather_data.groupby(['timestamp_hour']).agg({
                'temperature_c': 'mean',
                'humidity_pct': 'mean',
                'wind_speed_kmh': 'mean',
                'weather_category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
                'location': 'first'  # All values should be "Hanoi"
            }).reset_index()
            
            # Rename timestamp_hour to match air quality data's time_bucket column
            self.hourly_weather_data.rename(columns={'timestamp_hour': 'time_bucket'}, inplace=True)
            
            print(f"Successfully aggregated weather data to {len(self.hourly_weather_data)} hourly records.")
            return True
        except Exception as e:
            print(f"Error aggregating weather data: {e}")
            return False
    
    def merge_datasets(self):
        """
        Merge air quality and weather datasets based on timestamp.
        
        Returns:
            Boolean indicating if merge was successful
        """
        if self.raw_air_quality_data is None:
            print("No air quality data available. Call load_air_quality_data() first.")
            return False
            
        if self.hourly_weather_data is None:
            print("No hourly weather data available. Call aggregate_weather_to_hourly() first.")
            return False
        
        try:
            # Make copies to avoid modifying the original dataframes
            air_quality_df = self.raw_air_quality_data.copy()
            weather_df = self.hourly_weather_data.copy()
            # Remove 'location' from weather_df since it's always 'Hanoi'
            if 'location' in weather_df.columns:
                weather_df = weather_df.drop(columns=['location'])
                
            # Print column information for debugging
            print("Air quality columns:", air_quality_df.columns.tolist())
            print("Weather columns:", weather_df.columns.tolist())
            
            # Fix for duplicate 'location' column in air_quality_df
            if 'location' in air_quality_df.columns and air_quality_df.columns.tolist().count('location') > 1:
                # Keep only the first occurrence of 'location' column
                air_quality_df = air_quality_df.loc[:, ~air_quality_df.columns.duplicated()]
                print("Removed duplicate 'location' column from air quality data.")
                print("Updated air quality columns:", air_quality_df.columns.tolist())
            
            # Ensure time_bucket is datetime type in both dataframes
            air_quality_df['time_bucket'] = pd.to_datetime(air_quality_df['time_bucket'])
            weather_df['time_bucket'] = pd.to_datetime(weather_df['time_bucket'])
            
            # Remove any timezone info to ensure compatibility
            if hasattr(air_quality_df['time_bucket'].dt, 'tz') and air_quality_df['time_bucket'].dt.tz is not None:
                air_quality_df['time_bucket'] = air_quality_df['time_bucket'].dt.tz_localize(None)
                
            if hasattr(weather_df['time_bucket'].dt, 'tz') and weather_df['time_bucket'].dt.tz is not None:
                weather_df['time_bucket'] = weather_df['time_bucket'].dt.tz_localize(None)
            
            # Ensure air_quality_df is unique by location and time_bucket before merging
            if air_quality_df.duplicated(subset=['location', 'time_bucket']).any():
                # Count duplicates
                num_duplicates = air_quality_df.duplicated(subset=['location', 'time_bucket']).sum()
                print(f"Found {num_duplicates} duplicate (location, time_bucket) pairs in air quality data. Keeping first occurrences.")
                
                # Keep only the first occurrence of each (location, time_bucket) pair
                air_quality_df = air_quality_df.drop_duplicates(subset=['location', 'time_bucket'], keep='first')
            
            # Perform the merge
            print("Merging air quality and weather data...")
            merged_data = pd.merge(
                air_quality_df,
                weather_df,
                on='time_bucket',
                how='left'
            )
            
            print(f"Merged data shape: {merged_data.shape}")
            
            # Double-check for duplicate rows after merge
            duplicates = merged_data.duplicated(subset=['location', 'time_bucket'], keep='first')
            if duplicates.sum() > 0:
                print(f"Removing {duplicates.sum()} duplicate rows after merging...")
                merged_data = merged_data[~duplicates]
            
            # Process each location separately
            all_resampled_data = []
            
            # Get unique locations
            unique_locations = pd.Series(merged_data['location']).unique()
            print(f"Processing {len(unique_locations)} unique locations: {unique_locations}")
            
            for location in unique_locations:
                print(f"Processing location: {location}")
                
                # Filter data for this location
                location_data = merged_data[merged_data['location'] == location].copy()
                
                # Sort by time for proper resampling
                location_data = location_data.sort_values('time_bucket')
                
                # Get the min and max time for this location
                min_time = location_data['time_bucket'].min()
                max_time = location_data['time_bucket'].max()
                print(f"  Time range: {min_time} to {max_time}")
                
                # Create a complete hourly time range
                all_hours = pd.DataFrame({
                    'time_bucket': pd.date_range(start=min_time, end=max_time, freq='1H')
                })
                
                # Merge location data with the complete time range
                # This ensures we have a row for every hour
                location_hourly = pd.merge(
                    all_hours,
                    location_data,
                    on='time_bucket',
                    how='left'
                )
                
                # Fill the location column which might be missing after the merge
                location_hourly['location'] = location
                
                # Interpolate numeric columns
                numeric_cols = location_hourly.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    location_hourly[numeric_cols] = location_hourly[numeric_cols].interpolate(
                        method='linear', 
                        limit_direction='both'
                    )
                
                # Forward/backward fill non-numeric columns
                non_numeric_cols = [col for col in location_hourly.columns 
                                   if col not in numeric_cols and col != 'time_bucket']
                if non_numeric_cols:
                    location_hourly[non_numeric_cols] = location_hourly[non_numeric_cols].fillna(
                        method='ffill'
                    ).fillna(method='bfill')
                
                # Verify the uniqueness of time_bucket for this location
                if location_hourly.duplicated(subset=['time_bucket']).any():
                    print(f"Warning: Found duplicate timestamps for location {location} after resampling!")
                    location_hourly = location_hourly.drop_duplicates(subset=['time_bucket'], keep='first')
                
                # Add to our collection
                all_resampled_data.append(location_hourly)
            
            # Combine all the location data
            if all_resampled_data:
                final_data = pd.concat(all_resampled_data, ignore_index=False)
                print(f"Final combined data shape: {final_data.shape}")
                
                # Final verification of uniqueness by (location, time_bucket)
                duplicates = final_data.duplicated(subset=['location', 'time_bucket'])
                if duplicates.any():
                    print(f"Warning: Found {duplicates.sum()} duplicate (location, time_bucket) pairs in final data!")
                    final_data = final_data.drop_duplicates(subset=['location', 'time_bucket'], keep='first')
                    print(f"Removed duplicates. New shape: {final_data.shape}")
                else:
                    print("Final data is unique by (location, time_bucket) as expected.")
                
                # Set the time_bucket as index
                final_data = final_data.set_index('time_bucket')
                
                # Store the result
                self.data = final_data
                
                print(f"Successfully merged datasets with {len(self.data)} records.")
                print("Final columns:", self.data.columns.tolist())
                return True
            else:
                print("No data after processing.")
                return False
                
        except Exception as e:
            print(f"Error merging datasets: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_data(self):
        """
        Legacy method for backwards compatibility. Now loads from CSV.
        """
        try:
            if os.path.exists('hourly_air_quality_weather.csv'):
                self.data = pd.read_csv('hourly_air_quality_weather.csv')
                print(f"Successfully loaded data with {len(self.data)} records from CSV")
                
                # Convert time_bucket to datetime
                self.data['time_bucket'] = pd.to_datetime(self.data['time_bucket'])
                
                # Set time_bucket as index
                self.data = self.data.set_index('time_bucket')
                
                # Remove duplicate columns
                self.data = self.data.loc[:, ~self.data.columns.duplicated()]
                
                # Display basic info
                print(f"Data timespan: {self.data.index.min()} to {self.data.index.max()}")
                return True
            else:
                print("CSV file not found. Use extract_and_process_data() to create the dataset from raw sources.")
                return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def extract_and_process_data(self, start_date=None, end_date=None, save_to_csv=True):
        """
        Extract and process data from raw sources (TimescaleDB and weather CSV).
        This is a new method that combines all the data loading and processing steps.
        
        Args:
            start_date: Start date for air quality data extraction (defaults to 30 days ago)
            end_date: End date for air quality data extraction (defaults to now)
            save_to_csv: Whether to save the processed data to CSV
            
        Returns:
            Boolean indicating if processing was successful
        """
        try:
            # Step 1: Load air quality data from TimescaleDB
            print("Step 1: Loading air quality data from TimescaleDB...")
            if not self.load_air_quality_data(start_date, end_date):
                return False
                
            # Step 2: Load weather data from CSV
            print("Step 2: Loading weather data from CSV...")
            if not self.load_weather_data():
                return False
                
            # Step 3: Preprocess weather data
            print("Step 3: Preprocessing weather data...")
            if not self.preprocess_weather_data():
                return False
                
            # Step 4: Aggregate weather data to hourly intervals
            print("Step 4: Aggregating weather data to hourly intervals...")
            if not self.aggregate_weather_to_hourly():
                return False
                
            # Step 5: Merge datasets
            print("Step 5: Merging air quality and weather datasets...")
            if not self.merge_datasets():
                return False
                
            # Step 6: Optionally save to CSV
            if save_to_csv and self.data is not None:
                output_file = 'hourly_air_quality_weather.csv'
                self.data.reset_index().to_csv(output_file, index=False)
                print(f"Saved merged data to {output_file}")
                
                # Save as pickle too for faster loading
                pickle_file = 'hourly_air_quality_weather.pkl'
                self.data.reset_index().to_pickle(pickle_file)
                print(f"Saved merged data to {pickle_file} (pickle format)")
                
                # Save metadata
                metadata = {
                    'generated_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'num_records': len(self.data),
                    'time_range': f"{self.data.index.min()} to {self.data.index.max()}",
                    'locations': ', '.join(self.data['location'].unique() if 'location' in self.data.columns else ['Unknown']),
                    'processing_steps': [
                        "Extracted air quality data from TimescaleDB with station-level granularity",
                        "Loaded weather data from CSV and set all locations to \"Hanoi\"",
                        "Calculated AQI values for all pollutants",
                        "Merged datasets by timestamp",
                        "Resampled to hourly frequency with linear interpolation"
                    ]
                }
                
                with open('hourly_air_quality_weather_metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                print("Saved processing metadata to hourly_air_quality_weather_metadata.json")
            
            print("Data extraction and processing completed successfully.")
            return True
        except Exception as e:
            print(f"Error in data extraction and processing: {e}")
            return False

    def preprocess_data(self):
        """
        Preprocess the data, handling missing values and inappropriate data types.
        """
        if self.data is None:
            print("No data loaded. Call load_data() or extract_and_process_data() first.")
            return False
        
        # Handle missing values in key numerical columns
        numerical_cols = [
            'avg_pm25', 'avg_pm10', 'avg_o3', 'avg_no2', 'avg_so2', 'avg_co',
            'aqi_pm25', 'aqi_pm10', 'aqi_o3', 'aqi_co', 'aqi_so2', 'aqi_no2',
            'calculated_aqi', 'temperature_c', 'humidity_pct', 'wind_speed_kmh'
        ]
        
        for col in numerical_cols:
            if col in self.data.columns:
                # Interpolate missing values
                self.data[col] = self.data[col].interpolate(method='linear')
                # Fill any remaining NaNs at the edges
                self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
        
        # Convert categorical data
        if 'weather_category' in self.data.columns:
            self.data['weather_category'] = self.data['weather_category'].fillna('unknown')
        
        if 'aqi_category' in self.data.columns:
            self.data['aqi_category'] = self.data['aqi_category'].fillna('Unknown')
        
        # Check for any remaining NaNs
        missing_counts = self.data.isnull().sum()
        if missing_counts.sum() > 0:
            print("Remaining missing values:")
            print(missing_counts[missing_counts > 0])
            
        print("Preprocessing completed.")
        return True
    
    def exploratory_data_analysis(self):
        """
        Perform exploratory data analysis on the dataset.
        """
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        # Summary statistics
        print("Summary statistics:")
        summary_stats = self.data.describe()
        print(summary_stats)
        
        # Get unique locations
        if 'location' in self.data.columns:
            locations = self.data['location'].unique()
            print(f"Found {len(locations)} locations: {locations}")
        else:
            locations = ['All']
        
        # Time series plots for each location
        plt.figure(figsize=(14, 8))
        
        if 'location' in self.data.columns and len(locations) > 1:
            # Plot for each location with different colors
            for i, location in enumerate(locations):
                location_data = self.data[self.data['location'] == location]
                plt.plot(location_data.index, location_data['calculated_aqi'], 
                        marker='.', linestyle='-', alpha=0.7, label=f'Location: {location}')
        else:
            # Single plot if no location column or only one location
            plt.plot(self.data.index, self.data['calculated_aqi'], 
                    marker='.', linestyle='-', alpha=0.7, label='AQI')
        
        plt.title('AQI Time Series by Location')
        plt.xlabel('Date')
        plt.ylabel('AQI Value')
        plt.grid(True, alpha=0.3)
        
        # Add colored background for AQI categories
        for (lower, upper), category in self.aqi_categories.items():
            plt.axhspan(lower, upper, alpha=0.1, color=self.aqi_colors[category], label=category)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Distribution of AQI values by location
        if 'location' in self.data.columns and len(locations) > 1:
            plt.figure(figsize=(14, 8))
            for location in locations:
                location_data = self.data[self.data['location'] == location]
                sns.kdeplot(location_data['calculated_aqi'], label=f'Location: {location}')
            plt.title('Distribution of AQI Values by Location')
            plt.xlabel('AQI Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # Basic distribution of AQI values
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['calculated_aqi'], bins=20, kde=True)
        plt.title('Distribution of AQI Values')
        plt.xlabel('AQI Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap of numerical features
        plt.figure(figsize=(12, 10))
        numerical_data = self.data.select_dtypes(include=[np.number])
        correlation = numerical_data.corr()
        mask = np.triu(correlation)
        sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', 
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Weather feature relationships
        if all(col in self.data.columns for col in ['temperature_c', 'humidity_pct', 'wind_speed_kmh']):
            fig, axs = plt.subplots(3, 1, figsize=(12, 12))
            
            # Add location-based coloring if multiple locations exist
            if 'location' in self.data.columns and len(locations) > 1:
                for location in locations:
                    location_data = self.data[self.data['location'] == location]
                    axs[0].scatter(location_data['temperature_c'], location_data['calculated_aqi'], 
                                  alpha=0.6, label=f'Location: {location}')
                    axs[1].scatter(location_data['humidity_pct'], location_data['calculated_aqi'], 
                                  alpha=0.6, label=f'Location: {location}')
                    axs[2].scatter(location_data['wind_speed_kmh'], location_data['calculated_aqi'], 
                                  alpha=0.6, label=f'Location: {location}')
                axs[0].legend()
                axs[1].legend()
                axs[2].legend()
            else:
                axs[0].scatter(self.data['temperature_c'], self.data['calculated_aqi'], alpha=0.6)
                axs[1].scatter(self.data['humidity_pct'], self.data['calculated_aqi'], alpha=0.6)
                axs[2].scatter(self.data['wind_speed_kmh'], self.data['calculated_aqi'], alpha=0.6)
            
            axs[0].set_title('AQI vs Temperature')
            axs[0].set_xlabel('Temperature (°C)')
            axs[0].set_ylabel('AQI')
            axs[0].grid(True, alpha=0.3)
            
            axs[1].set_title('AQI vs Humidity')
            axs[1].set_xlabel('Humidity (%)')
            axs[1].set_ylabel('AQI')
            axs[1].grid(True, alpha=0.3)
            
            axs[2].set_title('AQI vs Wind Speed')
            axs[2].set_xlabel('Wind Speed (km/h)')
            axs[2].set_ylabel('AQI')
            axs[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Time-based patterns by location
        if len(self.data) >= 24:  # Need at least 24 hours of data
            # Hourly pattern
            if 'location' in self.data.columns and len(locations) > 1:
                plt.figure(figsize=(12, 8))
                for location in locations:
                    location_data = self.data[self.data['location'] == location]
                    hourly_pattern = location_data.groupby(location_data.index.hour)['calculated_aqi'].mean()
                    plt.plot(hourly_pattern.index, hourly_pattern.values, 
                            marker='o', linestyle='-', label=f'Location: {location}')
                plt.title('Average AQI by Hour of Day and Location')
                plt.legend()
            else:
                hourly_pattern = self.data.groupby(self.data.index.hour)['calculated_aqi'].mean()
                plt.figure(figsize=(10, 6))
                hourly_pattern.plot(kind='bar')
                plt.title('Average AQI by Hour of Day')
            
            plt.xlabel('Hour')
            plt.ylabel('Average AQI')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        if len(self.data) >= 168:  # Need at least a week of data
            # Daily pattern
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            if 'location' in self.data.columns and len(locations) > 1:
                plt.figure(figsize=(12, 8))
                for location in locations:
                    location_data = self.data[self.data['location'] == location]
                    daily_pattern = location_data.groupby(location_data.index.dayofweek)['calculated_aqi'].mean()
                    plt.plot(daily_pattern.index, daily_pattern.values, 
                            marker='o', linestyle='-', label=f'Location: {location}')
                plt.title('Average AQI by Day of Week and Location')
                plt.xticks(range(7), day_names)
                plt.legend()
            else:
                daily_pattern = self.data.groupby(self.data.index.dayofweek)['calculated_aqi'].mean()
                plt.figure(figsize=(10, 6))
                daily_pattern.plot(kind='bar')
                plt.title('Average AQI by Day of Week')
                plt.xticks(range(7), day_names)
            
            plt.xlabel('Day')
            plt.ylabel('Average AQI')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Box plots for AQI by day of week
            plt.figure(figsize=(14, 8))
            
            if 'location' in self.data.columns and len(locations) > 1:
                # Create a copy with day names for better plotting
                plot_data = self.data.copy()
                plot_data['day_of_week'] = plot_data.index.dayofweek.map(lambda x: day_names[x])
                
                # Create boxplot with location-based coloring
                sns.boxplot(x='day_of_week', y='calculated_aqi', hue='location', data=plot_data)
                plt.title('AQI Distribution by Day of Week and Location')
            else:
                self.data['day_of_week'] = self.data.index.dayofweek.map(lambda x: day_names[x])
                sns.boxplot(x='day_of_week', y='calculated_aqi', data=self.data)
                plt.title('AQI Distribution by Day of Week')
                
            plt.xlabel('Day of Week')
            plt.ylabel('AQI Value')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        print("Exploratory data analysis completed.")
    
    def prepare_data_for_forecasting(self, target_column='calculated_aqi'):
        """
        Prepare data for forecasting by creating time-based features and lagged variables.
        
        Args:
            target_column: The column to predict
            
        Returns:
            DataFrame ready for forecasting
        """
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return None
        
        # Create a copy of the data
        data = self.data.copy()
        
        # Select relevant features for forecasting
        features = [target_column]
        
        # Add weather features if available
        weather_features = ['temperature_c', 'humidity_pct', 'wind_speed_kmh']
        for feature in weather_features:
            if feature in data.columns and data[feature].notna().any():
                features.append(feature)
        
        # Add individual pollutant data if available
        pollutant_features = ['avg_pm25', 'avg_pm10', 'avg_o3', 'avg_no2', 'avg_so2', 'avg_co']
        for feature in pollutant_features:
            if feature in data.columns and data[feature].notna().any():
                features.append(feature)
        
        # Add location if available
        location_column = None
        if 'location' in data.columns:
            location_column = 'location'  # Store for later use, but don't add to features yet
            # Get unique locations
            locations = data['location'].unique()
            print(f"Found {len(locations)} locations: {locations}")
        
        # Create a subset with only the features we need
        forecast_data = data[features].copy()
        
        # Add time-based features
        forecast_data['hour'] = forecast_data.index.hour
        forecast_data['day_of_week'] = forecast_data.index.dayofweek
        forecast_data['month'] = forecast_data.index.month
        forecast_data['day'] = forecast_data.index.day
        forecast_data['is_weekend'] = forecast_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Add cyclical encoding for time features (better for ML models)
        forecast_data['hour_sin'] = np.sin(forecast_data['hour'] * (2 * np.pi / 24))
        forecast_data['hour_cos'] = np.cos(forecast_data['hour'] * (2 * np.pi / 24))
        forecast_data['day_of_week_sin'] = np.sin(forecast_data['day_of_week'] * (2 * np.pi / 7))
        forecast_data['day_of_week_cos'] = np.cos(forecast_data['day_of_week'] * (2 * np.pi / 7))
        forecast_data['month_sin'] = np.sin(forecast_data['month'] * (2 * np.pi / 12))
        forecast_data['month_cos'] = np.cos(forecast_data['month'] * (2 * np.pi / 12))
        
        # Add time periods typically associated with air quality patterns
        forecast_data['is_morning_rush'] = ((forecast_data['hour'] >= 7) & (forecast_data['hour'] <= 9)).astype(int)
        forecast_data['is_evening_rush'] = ((forecast_data['hour'] >= 17) & (forecast_data['hour'] <= 19)).astype(int)
        forecast_data['is_night'] = ((forecast_data['hour'] >= 22) | (forecast_data['hour'] <= 5)).astype(int)
        
        # Process each location separately if we have location data
        if location_column:
            # Initialize an empty dataframe to store processed data
            processed_data = []
            
            for location in locations:
                print(f"Preparing forecast data for location: {location}")
                # Filter data for this location
                location_data = data[data[location_column] == location].copy()
                
                # Get target and other numerical features
                loc_forecast_data = location_data[features].copy()
                
                # Add time-based features
                loc_forecast_data['hour'] = loc_forecast_data.index.hour
                loc_forecast_data['day_of_week'] = loc_forecast_data.index.dayofweek
                loc_forecast_data['month'] = loc_forecast_data.index.month
                loc_forecast_data['day'] = loc_forecast_data.index.day
                loc_forecast_data['is_weekend'] = loc_forecast_data['day_of_week'].isin([5, 6]).astype(int)
                
                # Add cyclical encoding
                loc_forecast_data['hour_sin'] = np.sin(loc_forecast_data['hour'] * (2 * np.pi / 24))
                loc_forecast_data['hour_cos'] = np.cos(loc_forecast_data['hour'] * (2 * np.pi / 24))
                loc_forecast_data['day_of_week_sin'] = np.sin(loc_forecast_data['day_of_week'] * (2 * np.pi / 7))
                loc_forecast_data['day_of_week_cos'] = np.cos(loc_forecast_data['day_of_week'] * (2 * np.pi / 7))
                
                # Add time periods
                loc_forecast_data['is_morning_rush'] = ((loc_forecast_data['hour'] >= 7) & (loc_forecast_data['hour'] <= 9)).astype(int)
                loc_forecast_data['is_evening_rush'] = ((loc_forecast_data['hour'] >= 17) & (loc_forecast_data['hour'] <= 19)).astype(int)
                loc_forecast_data['is_night'] = ((loc_forecast_data['hour'] >= 22) | (loc_forecast_data['hour'] <= 5)).astype(int)
                
                # Add lagged features (previous hours' values) for this location
                # Include more recent lags for short-term patterns
                for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:  # 1hr to 1 week
                    loc_forecast_data[f'{target_column}_lag_{lag}'] = loc_forecast_data[target_column].shift(lag)
                
                # Add rolling averages (moving averages) for this location
                for window in [3, 6, 12, 24, 48, 72]:
                    loc_forecast_data[f'{target_column}_rolling_{window}h'] = loc_forecast_data[target_column].rolling(window=window).mean()
                    # Add standard deviation to capture volatility
                    loc_forecast_data[f'{target_column}_rolling_{window}h_std'] = loc_forecast_data[target_column].rolling(window=window).std()
                
                # Add rate of change features
                loc_forecast_data[f'{target_column}_diff_1h'] = loc_forecast_data[target_column].diff()
                loc_forecast_data[f'{target_column}_diff_24h'] = loc_forecast_data[target_column].diff(24)
                
                # Add rolling minimum and maximum
                loc_forecast_data[f'{target_column}_rolling_24h_min'] = loc_forecast_data[target_column].rolling(window=24).min()
                loc_forecast_data[f'{target_column}_rolling_24h_max'] = loc_forecast_data[target_column].rolling(window=24).max()
                
                # Add daily patterns (average by hour of day)
                hourly_avg = loc_forecast_data.groupby(loc_forecast_data.index.hour)[target_column].transform('mean')
                loc_forecast_data[f'{target_column}_hour_avg'] = hourly_avg
                
                # Add weekly patterns (average by day of week)
                dow_avg = loc_forecast_data.groupby(loc_forecast_data.index.dayofweek)[target_column].transform('mean')
                loc_forecast_data[f'{target_column}_dow_avg'] = dow_avg
                
                # Add location identifier
                loc_forecast_data['location'] = location
                
                # Add to our collection
                processed_data.append(loc_forecast_data)
            
            # Combine all processed location data
            forecast_data = pd.concat(processed_data)
            
            # One-hot encode the location column to create numerical features
            try:
                # Try with newer scikit-learn version
                encoder = OneHotEncoder(sparse_output=False, drop='first')
            except TypeError:
                try:
                    # Fallback for older scikit-learn versions
                    encoder = OneHotEncoder(sparse=False, drop='first')
                except:
                    # Fallback for very old versions
                    encoder = OneHotEncoder(drop='first')
                
            location_encoded = encoder.fit_transform(forecast_data[['location']])
            
            # Handle sparse matrix if needed
            if hasattr(location_encoded, 'toarray'):
                location_encoded = location_encoded.toarray()
            
            # Create a DataFrame with encoded location features
            location_encoded_df = pd.DataFrame(
                location_encoded, 
                columns=[f'location_{loc}' for loc in encoder.categories_[0][1:]], 
                index=forecast_data.index
            )
            
            # Drop the original string location column and add encoded columns
            forecast_data = forecast_data.drop(columns=['location'])
            forecast_data = pd.concat([forecast_data, location_encoded_df], axis=1)
            
            print(f"Encoded locations as {len(encoder.categories_[0])-1} binary features")
            
        else:
            # Process for single location (or combined) data
            
            # Add lagged features (previous hours' values)
            for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:  # 1hr to 1 week
                forecast_data[f'{target_column}_lag_{lag}'] = forecast_data[target_column].shift(lag)
            
            # Add rolling averages (moving averages) with different windows
            for window in [3, 6, 12, 24, 48, 72]:
                forecast_data[f'{target_column}_rolling_{window}h'] = forecast_data[target_column].rolling(window=window).mean()
                # Add standard deviation to capture volatility
                forecast_data[f'{target_column}_rolling_{window}h_std'] = forecast_data[target_column].rolling(window=window).std()
            
            # Add rolling min/max for recent values
            forecast_data[f'{target_column}_rolling_24h_min'] = forecast_data[target_column].rolling(window=24).min()
            forecast_data[f'{target_column}_rolling_24h_max'] = forecast_data[target_column].rolling(window=24).max()
                
            # Add rate of change features
            forecast_data[f'{target_column}_diff_1h'] = forecast_data[target_column].diff()
            forecast_data[f'{target_column}_diff_24h'] = forecast_data[target_column].diff(24)
            
            # Add daily patterns (average by hour of day)
            hourly_avg = forecast_data.groupby(forecast_data.index.hour)[target_column].transform('mean')
            forecast_data[f'{target_column}_hour_avg'] = hourly_avg
            
            # Add weekly patterns (average by day of week)
            dow_avg = forecast_data.groupby(forecast_data.index.dayofweek)[target_column].transform('mean')
            forecast_data[f'{target_column}_dow_avg'] = dow_avg
        
        # Try to add seasonal decomposition if the data is sufficient
        if len(forecast_data) >= 48:  # Need at least two days of data
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Create a temporary series for decomposition
                # (need continuous time series without gaps)
                temp_series = forecast_data[target_column].copy()
                
                # Fill any missing values for decomposition
                temp_series = temp_series.interpolate(method='linear')
                
                # Get decomposition with 24 hour seasonality
                try:
                    decomposition = seasonal_decompose(temp_series, period=24, model='additive')
                    
                    # Add components to the data
                    forecast_data[f'{target_column}_trend'] = decomposition.trend
                    forecast_data[f'{target_column}_seasonal'] = decomposition.seasonal
                    forecast_data[f'{target_column}_resid'] = decomposition.resid
                    
                    print("Added seasonal decomposition features")
                except Exception as e:
                    print(f"Could not perform seasonal decomposition: {e}")
            except ImportError:
                print("statsmodels seasonal_decompose not available, skipping decomposition")
        
        # Add interaction features between weather and time
        if 'temperature_c' in forecast_data.columns and 'hour' in forecast_data.columns:
            forecast_data['temp_hour_interaction'] = forecast_data['temperature_c'] * forecast_data['hour']
            
        if 'humidity_pct' in forecast_data.columns and 'hour' in forecast_data.columns:
            forecast_data['humidity_hour_interaction'] = forecast_data['humidity_pct'] * forecast_data['hour']
            
        # Add pollutant ratios if available
        if 'avg_pm25' in forecast_data.columns and 'avg_pm10' in forecast_data.columns:
            # Add small epsilon to avoid division by zero
            forecast_data['pm25_pm10_ratio'] = forecast_data['avg_pm25'] / (forecast_data['avg_pm10'] + 0.001)
            
        # Drop rows with NaN values (from lag creation)
        forecast_data = forecast_data.dropna()
        
        print(f"Prepared time series data with {len(forecast_data)} records and {len(forecast_data.columns)} features")
        
        # Store processed data
        self.forecast_data = forecast_data
        
        # Show correlation with target variable
        corr = forecast_data.corr()[target_column].sort_values(ascending=False)
        print("\nFeatures most correlated with target:")
        print(corr.head(15))
        
        return forecast_data
    
    def train_test_split(self, test_size=0.2, target_column='calculated_aqi'):
        """
        Split data into training and testing sets preserving time order.
        
        Args:
            test_size: Proportion of data to use for testing
            target_column: Target column to predict
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.forecast_data is None:
            print("No forecast data prepared. Call prepare_data_for_forecasting() first.")
            return None, None, None, None, None
        
        # Make a copy of the data
        df = self.forecast_data.copy()
        
        # Define feature columns (all columns except the target)
        # Importantly, exclude any direct leakage from target variable
        feature_columns = []
        for col in df.columns:
            # Exclude target column and any identical columns
            if col == target_column:
                continue
            # Check if this is a raw version of target or direct transformation
            if col.startswith(f"{target_column}_") and ("lag" not in col and "rolling" not in col):
                # Skip direct transformations of target
                continue
            else:
                feature_columns.append(col)
        
        # Drop any columns that might cause data leakage 
        # (e.g., future information or direct transformations of target)
        leakage_columns = [col for col in df.columns if f"{target_column}_diff" in col]
        for col in leakage_columns:
            if col in feature_columns:
                feature_columns.remove(col)
                print(f"Removed potential leakage column: {col}")
        
        # Sort data by timestamp to ensure chronological order
        df = df.sort_index()
        
        # Print data shape before splitting
        print(f"Original data shape: {df.shape} from {df.index.min()} to {df.index.max()}")
        
        # Split based on time (not randomly)
        train_size = int(len(df) * (1 - test_size))
        
        # Ensure we have enough data for training and testing
        if train_size < 10:
            print(f"Warning: Training set too small ({train_size} records). Using first 80% for training.")
            train_size = int(len(df) * 0.8)
        
        # Get training and testing data based on chronological order
        train_data = df.iloc[:train_size].copy()
        test_data = df.iloc[train_size:].copy()
        
        # Reset indices to avoid alignment issues
        train_data_indices = train_data.index
        test_data_indices = test_data.index
        
        # Store the test index for plotting
        test_index = test_data.index
        
        # Split into features and target for training data
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        
        # Split into features and target for testing data
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        # Convert y_train and y_test from pandas Series to numpy arrays to avoid index alignment issues
        # But keep original indices for plots
        y_train_values = y_train.values
        y_test_values = y_test.values
        
        # Normalize numerical features
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        self.scaler = StandardScaler()
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
        X_test_scaled[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        print(f"Training data: {len(X_train)} records from {train_data.index.min()} to {train_data.index.max()}")
        print(f"Testing data: {len(X_test)} records from {test_data.index.min()} to {test_data.index.max()}")
        print(f"Features used: {len(feature_columns)}")
        
        return X_train_scaled, X_test_scaled, y_train_values, y_test_values, test_index
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model predictions with comprehensive metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert inputs to numpy arrays if they're pandas Series
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
            
        # Ensure arrays have the same shape
        if len(y_true) != len(y_pred):
            print(f"Error: Length mismatch between y_true ({len(y_true)}) and y_pred ({len(y_pred)})")
            # Try to match lengths by taking the minimum length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            print(f"Truncated arrays to {min_len} elements")
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.nan
            
        # Calculate AQI category accuracy (important for health warnings)
        y_true_categories = np.array([self.get_aqi_category(val) for val in y_true])
        y_pred_categories = np.array([self.get_aqi_category(val) for val in y_pred])
        category_accuracy = np.mean(y_true_categories == y_pred_categories) * 100
        
        # Calculate directional accuracy (if AQI is increasing or decreasing)
        # This is important for trend prediction
        if len(y_true) > 1:
            y_true_direction = np.sign(np.diff(y_true))
            y_pred_direction = np.sign(np.diff(y_pred))
            direction_match = (y_true_direction == y_pred_direction)
            directional_accuracy = np.mean(direction_match) * 100
        else:
            directional_accuracy = np.nan
            
        # Calculate error by AQI category
        category_errors = {}
        unique_categories = np.unique(y_true_categories)
        for category in unique_categories:
            mask = y_true_categories == category
            if mask.sum() > 0:
                category_errors[category] = {
                    'count': mask.sum(),
                    'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                    'rmse': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                }
        
        print(f"{model_name} Performance:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Category Accuracy: {category_accuracy:.2f}%")
        print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        # Report category-specific performance
        if category_errors:
            print("\n  Performance by AQI Category:")
            for category, metrics in category_errors.items():
                print(f"    {category} (n={metrics['count']}): MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        
        # Comprehensive result dictionary
        result = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'category_accuracy': category_accuracy,
            'directional_accuracy': directional_accuracy,
            'category_errors': category_errors
        }
        
        self.model_results.append(result)
        return result
        
    def plot_predictions(self, y_true, y_pred, model_name, test_index, forecast_horizon=None):
        """
        Plot actual vs predicted values with more comprehensive visualization.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            test_index: Index for the test data
            forecast_horizon: Forecast horizon (e.g., 'Next Hour')
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot time series on top subplot
        ax1.plot(test_index, y_true, label='Actual', marker='o', alpha=0.6, markersize=4)
        ax1.plot(test_index, y_pred, label=f'Predicted ({model_name})', marker='x', linestyle='--', markersize=4)
        
        title = f'Actual vs Predicted AQI Values - {model_name}'
        if forecast_horizon:
            title += f" ({forecast_horizon})"
        
        ax1.set_title(title, fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('AQI Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add AQI category thresholds on first subplot
        for (lower, upper), category in self.aqi_categories.items():
            ax1.axhspan(lower, upper, alpha=0.1, color=self.aqi_colors[category])
            # Add category labels on the right side
            ax1.text(test_index[-1], (lower + upper) / 2, category, va='center', fontsize=8)
        
        # Plot prediction errors on bottom subplot
        errors = y_pred - y_true
        ax2.bar(test_index, errors, alpha=0.6, color='coral')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_title('Prediction Errors', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error (Predicted - Actual)')
        ax2.grid(True, alpha=0.3)
        
        # Add error statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax2.text(0.02, 0.95, f'Mean Error: {mean_error:.2f}\nStd Dev: {std_error:.2f}', 
                transform=ax2.transAxes, va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        # Add scatterplot of actual vs predicted values
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.title(f'Actual vs Predicted Values - {model_name}', fontsize=14)
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.grid(True, alpha=0.3)
        
        # Add quality metrics to the plot
        text = f'MAE: {mean_absolute_error(y_true, y_pred):.2f}\n'
        text += f'RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}\n'
        text += f'R²: {r2_score(y_true, y_pred):.4f}'
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, 
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        # Plot confusion matrix for AQI categories
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.colors as mcolors
            
            # Convert to AQI categories
            y_true_cat = pd.Series(y_true).apply(self.get_aqi_category)
            y_pred_cat = pd.Series(y_pred).apply(self.get_aqi_category)
            
            # Get unique categories present in the data
            categories = np.unique(np.concatenate([y_true_cat, y_pred_cat]))
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true_cat, y_pred_cat, labels=categories)
            
            # Plot
            plt.figure(figsize=(10, 8))
            # Normalize by row (true categories)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # Handle div by zero by replacing NaNs with zeros
            cm_norm = np.nan_to_num(cm_norm)
            
            # Create a colormap with white for 0 and red for 1
            cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"])
            
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap=cmap, 
                        xticklabels=categories, yticklabels=categories, vmin=0, vmax=1)
            plt.title(f'AQI Category Confusion Matrix - {model_name}', fontsize=14)
            plt.ylabel('True Category')
            plt.xlabel('Predicted Category')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Couldn't plot confusion matrix: {e}")
            
    def compare_models(self):
        """
        Compare the performance of different models with comprehensive metrics.
        """
        if not self.model_results:
            print("No model results available.")
            return
        
        # Make sure we have unique model names
        seen_models = set()
        unique_results = []
        
        for result in self.model_results:
            if result['model'] not in seen_models:
                seen_models.add(result['model'])
                unique_results.append(result)
        
        # Replace with deduplicated results
        results_df = pd.DataFrame(unique_results)
        
        # Extract numerical columns for plotting
        metric_cols = ['mae', 'rmse', 'r2', 'mape', 'category_accuracy', 'directional_accuracy']
        available_metrics = [col for col in metric_cols if col in results_df.columns]
        
        # Create a subset with just the metrics we need
        plot_df = results_df[['model'] + available_metrics].copy()
        
        # Sort by RMSE (lower is better)
        plot_df = plot_df.sort_values('rmse')
        
        print("\nModel Comparison:")
        print(plot_df[['model'] + available_metrics].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        # Determine color scheme for model types
        model_types = {
            'Linear': 'royalblue',
            'Ridge': 'lightblue',
            'Random Forest': 'green',
            'Gradient Boosting': 'limegreen',
            'XGBoost': 'forestgreen',
            'ARIMA': 'orangered',
            'SARIMAX': 'tomato',
            'VAR': 'darkorange',
            'Prophet': 'purple'
        }
        
        # Assign colors to models
        colors = []
        for model_name in plot_df['model']:
            # Find matching model type
            for model_type, color in model_types.items():
                if model_type in model_name:
                    colors.append(color)
                    break
            else:
                # Default color if no match
                colors.append('gray')
        
        # Get model names for plotting
        model_names = plot_df['model'].tolist()
        
        # Visualize comparison
        plt.figure(figsize=(14, 12))
        
        # Plot MAE
        plt.subplot(3, 1, 1)
        plt.barh(range(len(model_names)), plot_df['mae'], color=colors)
        plt.yticks(range(len(model_names)), model_names)
        plt.title('Mean Absolute Error (Lower is Better)', fontsize=14)
        plt.xlabel('MAE')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Plot RMSE
        plt.subplot(3, 1, 2)
        plt.barh(range(len(model_names)), plot_df['rmse'], color=colors)
        plt.yticks(range(len(model_names)), model_names)
        plt.title('Root Mean Squared Error (Lower is Better)', fontsize=14)
        plt.xlabel('RMSE')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Plot R² or category accuracy depending on what's available
        if 'r2' in available_metrics:
            plt.subplot(3, 1, 3)
            r2_vals = plot_df['r2']
            plt.barh(range(len(model_names)), r2_vals, color=colors)
            plt.yticks(range(len(model_names)), model_names)
            plt.title('R² Score (Higher is Better)', fontsize=14)
            plt.xlabel('R²')
            
            # Add negative R² warning annotation if applicable
            if (r2_vals < 0).any():
                neg_models = plot_df[plot_df['r2'] < 0]['model'].tolist()
                plt.annotate(f"Note: Negative R² indicates poor fit for: {', '.join(neg_models)}",
                            xy=(0.5, -0.1), xycoords='axes fraction', ha='center', 
                            fontsize=10, color='red')
        elif 'category_accuracy' in available_metrics:
            plt.subplot(3, 1, 3)
            plt.barh(range(len(model_names)), plot_df['category_accuracy'], color=colors)
            plt.yticks(range(len(model_names)), model_names)
            plt.title('AQI Category Prediction Accuracy (Higher is Better)', fontsize=14)
            plt.xlabel('Accuracy (%)')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()
        
        # Plot additional metrics if available
        if 'mape' in available_metrics and 'directional_accuracy' in available_metrics:
            plt.figure(figsize=(14, 10))
            
            # Plot MAPE
            plt.subplot(2, 1, 1)
            plt.barh(range(len(model_names)), plot_df['mape'], color=colors)
            plt.yticks(range(len(model_names)), model_names)
            plt.title('Mean Absolute Percentage Error (Lower is Better)', fontsize=14)
            plt.xlabel('MAPE (%)')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Plot directional accuracy
            plt.subplot(2, 1, 2)
            plt.barh(range(len(model_names)), plot_df['directional_accuracy'], color=colors)
            plt.yticks(range(len(model_names)), model_names)
            plt.title('Directional Accuracy (Higher is Better)', fontsize=14)
            plt.xlabel('Accuracy (%)')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        
        # Create radar chart for model comparison
        try:
            # Metrics to include in radar chart
            radar_metrics = ['mae', 'rmse', 'mape', 'category_accuracy', 'directional_accuracy']
            available_radar_metrics = [m for m in radar_metrics if m in available_metrics]
            
            if len(available_radar_metrics) >= 3:  # Need at least 3 metrics for radar chart
                # Normalize metrics for radar chart (0-1 scale, where 1 is best)
                radar_df = plot_df[['model'] + available_radar_metrics].copy()
                
                # For metrics where lower is better (mae, rmse, mape), invert the scale
                for metric in ['mae', 'rmse', 'mape']:
                    if metric in radar_df.columns:
                        max_val = radar_df[metric].max()
                        if max_val > 0:
                            radar_df[metric] = 1 - (radar_df[metric] / max_val)
                
                # For metrics where higher is better, normalize to 0-1
                for metric in ['category_accuracy', 'directional_accuracy', 'r2']:
                    if metric in radar_df.columns:
                        min_val = radar_df[metric].min()
                        max_val = radar_df[metric].max()
                        if max_val > min_val:
                            radar_df[metric] = (radar_df[metric] - min_val) / (max_val - min_val)
                
                # Cap values at 0 and 1
                for metric in available_radar_metrics:
                    radar_df[metric] = radar_df[metric].clip(0, 1)
                
                # Prepare radar chart
                # Get top 5 models based on RMSE
                top_models = radar_df.head(5)['model'].tolist()
                
                # Prepare angle and metrics
                angles = np.linspace(0, 2*np.pi, len(available_radar_metrics), endpoint=False).tolist()
                angles += angles[:1]  # Close the loop
                
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                
                # Plot each model
                for i, model in enumerate(top_models):
                    values = radar_df[radar_df['model'] == model][available_radar_metrics].values.flatten().tolist()
                    values += values[:1]  # Close the loop
                    
                    ax.plot(angles, values, linewidth=2, label=model, color=colors[plot_df['model'].tolist().index(model)])
                    ax.fill(angles, values, alpha=0.1, color=colors[plot_df['model'].tolist().index(model)])
                
                # Add labels
                metric_labels = [m.upper() if m != 'r2' else 'R²' for m in available_radar_metrics]
                metric_labels += [metric_labels[0]]  # Close the loop
                
                ax.set_thetagrids(np.degrees(angles), metric_labels)
                
                # Configure plot
                ax.set_ylim(0, 1)
                ax.set_yticks([0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
                ax.grid(True)
                
                plt.title("Model Comparison (Normalized Metrics, Higher is Better)", fontsize=15)
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Couldn't create radar chart: {e}")
        
        # Return the best model based on RMSE
        best_model = plot_df.iloc[0]['model']
        print(f"\nBest performing model based on RMSE: {best_model}")
        return best_model
    
    def train_arima_model(self, y_train, y_test, test_index):
        """
        Train an ARIMA model for univariate time series forecasting.
        
        Args:
            y_train: Training target values
            y_test: Testing target values
            test_index: Index for test data
            
        Returns:
            Evaluation results or None if model is determined to be unsuitable
        """
        try:
            print("\nTraining ARIMA model...")
            
            # Handle short time series
            if len(y_train) < 24:  # Increased minimum requirement
                print(f"Warning: Training data too short for ARIMA ({len(y_train)} points). Needs at least 24 points.")
                print("Skipping ARIMA model.")
                return None
                
            # Convert to numpy array if pandas Series
            if hasattr(y_train, 'values'):
                y_train_values = y_train.values
            else:
                y_train_values = y_train
            
            # Get statistics for scaling and validation
            train_mean = np.mean(y_train_values)
            train_std = np.std(y_train_values)
            train_min = np.min(y_train_values)
            train_max = np.max(y_train_values)
            
            # Define reasonable bounds based on training data
            lower_bound = max(0, train_min - 2 * train_std)
            upper_bound = train_max + 3 * train_std
            
            print(f"Training data statistics: mean={train_mean:.2f}, std={train_std:.2f}, min={train_min:.2f}, max={train_max:.2f}")
            print(f"Setting prediction bounds to: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Handle outliers in training data
            train_filtered_values = y_train_values.copy()
            outlier_mask = (train_filtered_values > upper_bound) | (train_filtered_values < lower_bound)
            if outlier_mask.any():
                num_outliers = np.sum(outlier_mask)
                print(f"Detecting and handling {num_outliers} outliers in training data")
                # Replace outliers with the median of neighboring points
                for i in np.where(outlier_mask)[0]:
                    if i > 0 and i < len(train_filtered_values) - 1:
                        train_filtered_values[i] = np.median(train_filtered_values[max(0, i-2):min(len(train_filtered_values), i+3)])
                    else:
                        train_filtered_values[i] = train_mean
            
            # Check stationarity
            result = adfuller(train_filtered_values)
            print(f'ADF Statistic: {result[0]:.4f}')
            print(f'p-value: {result[1]:.4f}')
            
            is_stationary = result[1] < 0.05
            print(f"Series is {'stationary' if is_stationary else 'non-stationary'}")
            
            # If non-stationary, take first difference
            if not is_stationary and len(train_filtered_values) > 24:  # Increased minimum requirement
                print("Taking first difference to achieve stationarity...")
                y_train_diff = np.diff(train_filtered_values)
                if len(y_train_diff) > 0:
                    result_diff = adfuller(y_train_diff)
                    print(f'ADF Statistic after differencing: {result_diff[0]:.4f}')
                    print(f'p-value after differencing: {result_diff[1]:.4f}')
                    diff_order = 1
                else:
                    print("Not enough data after differencing. Using d=0.")
                    diff_order = 0
            else:
                diff_order = 0
            
            # Try several possible ARIMA orders to find the best one
            try:
                from pmdarima import auto_arima
                
                # Find best parameters with conservative settings
                print("Finding optimal ARIMA parameters...")
                arima_model = auto_arima(
                    train_filtered_values,
                    d=diff_order,
                    start_p=0, start_q=0,
                    max_p=2, max_q=2,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_order=4,  # Limit total order
                    information_criterion='aic',  # Use AIC for model selection
                    trace=False
                )
                
                order = arima_model.order
                print(f"Best ARIMA order: {order}")
                
            except ImportError:
                # Default parameters if auto_arima is not available
                if diff_order == 1:
                    order = (1, 1, 0)  # AR(1) with differencing, no MA term
                else:
                    order = (1, 0, 0)  # Simple AR(1) model
                print(f"Using default ARIMA order: {order}")
            except Exception as e:
                print(f"Error in auto_arima: {e}")
                # Fallback to simple model
                if diff_order == 1:
                    order = (1, 1, 0)
                else:
                    order = (1, 0, 0)
                print(f"Using fallback ARIMA order: {order}")
            
            # Check if model is not too complex for the data
            total_order = sum(order)
            if total_order > len(train_filtered_values) / 10:
                print(f"Warning: ARIMA order {order} may be too complex for dataset size {len(train_filtered_values)}")
                # Simplify the model
                if diff_order == 1:
                    order = (1, 1, 0)
                else:
                    order = (1, 0, 0)
                print(f"Simplifying to ARIMA{order}")
            
            # Train ARIMA model
            arima = ARIMA(train_filtered_values, order=order)
            arima_fit = arima.fit()
            
            # Use summary for diagnostics
            print("ARIMA model summary:")
            print(arima_fit.summary().tables[0].as_text())
            
            # Make predictions
            y_pred = arima_fit.forecast(steps=len(y_test))
            
            # Convert to numpy array if a pandas Series
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            
            # Check for unreasonable predictions and clip them
            original_pred = y_pred.copy()
            y_pred = np.clip(y_pred, lower_bound, upper_bound)
            
            # Count how many predictions needed to be clipped
            num_clipped = np.sum(original_pred != y_pred)
            if num_clipped > 0:
                print(f"Warning: {num_clipped} out of {len(y_pred)} predictions were clipped to be within reasonable bounds")
                
            # Ensure non-negative predictions
            y_pred = np.maximum(y_pred, 0)
            
            # Check if predictions are wildly off
            pred_mean = np.mean(y_pred)
            pred_std = np.std(y_pred)
            
            # Skip model if the predictions are unreasonable
            if pred_mean > 5 * train_mean or pred_std > 5 * train_std:
                print(f"Warning: ARIMA predictions are unreasonable (mean={pred_mean:.2f}, std={pred_std:.2f})")
                print("Skipping ARIMA model in comparison to avoid distorting graphs")
                return None
            
            # Evaluate
            model_name = f"ARIMA{order}"
            result = self.evaluate_model(y_test, y_pred, model_name)
            
            # Only proceed if results are reasonable
            if result['rmse'] > 5 * train_std:
                print(f"Warning: ARIMA model has very high RMSE ({result['rmse']:.2f}) compared to data standard deviation ({train_std:.2f})")
                print("Skipping ARIMA model in comparison to avoid distorting graphs")
                # Remove from model_results to avoid affecting the graphs
                if hasattr(self, 'model_results'):
                    self.model_results = [r for r in self.model_results if r['model'] != model_name]
                return None
            
            # Plot only if results are reasonable
            self.plot_predictions(y_test, y_pred, model_name, test_index)
            
            # Store the model
            self.models[model_name] = arima_fit
            
            return result
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_prophet_model(self, y_train, y_test, test_index):
        """
        Train a Prophet model for time series forecasting.
        
        Args:
            y_train: Training target values
            y_test: Testing target values
            test_index: Index for test data
            
        Returns:
            Evaluation results
        """
        try:
            # Import Prophet
            from prophet import Prophet
            
            print("\nTraining Prophet model...")
            
            # Handle short time series
            if len(y_train) < 30:
                print(f"Warning: Training data too short for Prophet ({len(y_train)} points). Needs at least 30 points.")
                print("Skipping Prophet model.")
                return None
            
            # Ensure y_train and y_test are numpy arrays
            if hasattr(y_train, 'values'):
                y_train_values = y_train.values
            else:
                y_train_values = y_train
                
            if hasattr(y_test, 'values'):
                y_test_values = y_test.values
            else:
                y_test_values = y_test
            
            # Get statistics for scaling and validation
            train_mean = np.mean(y_train_values)
            train_std = np.std(y_train_values)
            train_min = np.min(y_train_values)
            train_max = np.max(y_train_values)
            
            # Define reasonable bounds based on training data
            lower_bound = max(0, train_min - 2 * train_std)
            upper_bound = train_max + 3 * train_std
            
            print(f"Training data statistics: mean={train_mean:.2f}, std={train_std:.2f}, min={train_min:.2f}, max={train_max:.2f}")
            print(f"Setting prediction bounds to: [{lower_bound:.2f}, {upper_bound:.2f}]")
                
            # Create proper datetime indices for Prophet
            if not isinstance(test_index, pd.DatetimeIndex):
                print("Warning: test_index is not a DatetimeIndex. Creating synthetic dates.")
                train_ds = pd.date_range(
                    start='2020-01-01', 
                    periods=len(y_train_values), 
                    freq='H'
                )
                test_ds = pd.date_range(
                    start=train_ds[-1] + pd.Timedelta(hours=1), 
                    periods=len(y_test_values), 
                    freq='H'
                )
            else:
                # Calculate training dates based on test dates
                latest_test_date = test_index[0] - pd.Timedelta(hours=1)
                train_ds = pd.date_range(
                    end=latest_test_date,
                    periods=len(y_train_values),
                    freq='H'
                )
                test_ds = test_index
            
            # Handle outliers in training data
            train_filtered_values = y_train_values.copy()
            outlier_mask = (train_filtered_values > upper_bound) | (train_filtered_values < lower_bound)
            if outlier_mask.any():
                num_outliers = np.sum(outlier_mask)
                print(f"Detecting and interpolating {num_outliers} outliers in training data")
                train_filtered_values[outlier_mask] = np.clip(train_filtered_values[outlier_mask], lower_bound, upper_bound)
            
            # Create dataframe for Prophet with required column names
            train_data = pd.DataFrame({
                'ds': train_ds,
                'y': train_filtered_values
            })
            
            # Create test dataframe
            test_data = pd.DataFrame({
                'ds': test_ds,
                'y': y_test_values
            })
            
            # Check for missing or negative values (Prophet can't handle these)
            if np.isnan(train_data['y']).any() or (train_data['y'] < 0).any():
                print("Warning: Found NaN or negative values in training data. Fixing these...")
                train_data['y'] = np.nan_to_num(train_data['y'], nan=train_mean)
                train_data['y'] = np.maximum(train_data['y'], 0.1)  # Ensure positive values
            
            # Verify data is correct
            print(f"Training data shape: {train_data.shape}, Test data shape: {test_data.shape}")
            
            # Initialize and train Prophet model with more conservative hyperparameters
            prophet_model = Prophet(
                yearly_seasonality=False,  # Unless we have multiple years of data
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='additive',  # Changed from multiplicative to reduce extreme values
                interval_width=0.95,
                changepoint_prior_scale=0.05,  # Default is 0.05, lower to reduce flexibility
                seasonality_prior_scale=10.0,  # Default is 10, adjust based on known seasonality
                changepoint_range=0.8,        # Default is 0.8
                mcmc_samples=0               # Disable MCMC for faster fitting
            )
            
            # Add hourly seasonality (important for air quality prediction)
            prophet_model.add_seasonality(name='hourly', period=24, fourier_order=5, prior_scale=5.0)
            
            # Add special seasonality for rush hour patterns with reduced prior scale
            prophet_model.add_seasonality(name='rush_hour', period=24, fourier_order=3, prior_scale=3.0)
            
            # Fit the model
            prophet_model.fit(train_data)
            
            # Create future dataframe for the test period
            future = pd.DataFrame({'ds': test_ds})
            
            # Generate predictions
            forecast = prophet_model.predict(future)
            
            # Extract predictions for the test period
            y_pred = forecast['yhat'].values
            
            # Check for unreasonable predictions and clip them
            original_pred = y_pred.copy()
            y_pred = np.clip(y_pred, lower_bound, upper_bound)
            
            # Count how many predictions needed to be clipped
            num_clipped = np.sum(original_pred != y_pred)
            if num_clipped > 0:
                print(f"Warning: {num_clipped} out of {len(y_pred)} predictions were clipped to be within reasonable bounds")
            
            # Ensure non-negative predictions
            y_pred = np.maximum(y_pred, 0)
            
            # Check if predictions are wildly off
            pred_mean = np.mean(y_pred)
            pred_std = np.std(y_pred)
            
            # Skip model if the predictions are unreasonable
            if pred_mean > 5 * train_mean or pred_std > 5 * train_std:
                print(f"Warning: Prophet predictions are unreasonable (mean={pred_mean:.2f}, std={pred_std:.2f})")
                print("Skipping Prophet model in comparison to avoid distorting graphs")
                return None
            
            # Evaluate
            model_name = "Prophet"
            result = self.evaluate_model(y_test_values, y_pred, model_name)
            
            # Only proceed if results are reasonable
            if 'rmse' in result and result['rmse'] > 5 * train_std:
                print(f"Warning: Prophet model has very high RMSE ({result['rmse']:.2f}) compared to data standard deviation ({train_std:.2f})")
                print("Skipping Prophet model in comparison to avoid distorting graphs")
                # Remove from model_results to avoid affecting the graphs
                if hasattr(self, 'model_results'):
                    self.model_results = [r for r in self.model_results if r['model'] != model_name]
                return None
                
            # Plot predictions only if results are reasonable
            self.plot_predictions(y_test_values, y_pred, model_name, test_index)
            
            # Plot Prophet components (trend, seasonality) if results are reasonable
            try:
                fig = prophet_model.plot_components(forecast)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not plot Prophet components: {e}")
            
            # Store the model
            self.models[model_name] = prophet_model
            
            return result
            
        except ImportError:
            print("Prophet package not installed. Skipping Prophet model.")
            print("To install: pip install prophet")
            return None
        except Exception as e:
            print(f"Error training Prophet model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_sarimax_model(self, y_train, exog_train, y_test, exog_test, test_index):
        """
        Train a SARIMAX (Seasonal ARIMA with exogenous variables) model.
        
        Args:
            y_train: Training target values
            exog_train: Training exogenous variables 
            y_test: Testing target values
            exog_test: Testing exogenous variables
            test_index: Index for test data
            
        Returns:
            Evaluation results
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from pmdarima import auto_arima
            
            print("\nTraining SARIMAX model...")
            
            # Handle short time series
            if len(y_train) < 30:
                print(f"Warning: Training data too short for SARIMAX ({len(y_train)} points). Needs at least 30 points.")
                print("Skipping SARIMAX model.")
                return None
                
            # Ensure we have enough data for seasonal differencing
            # For a seasonal period of 24, we need at least 24*2 data points
            if len(y_train) < 48:
                print("Warning: Training data too short for seasonal differencing. Using non-seasonal ARIMA instead.")
                seasonal = False
                m = 1  # No seasonality
            else:
                seasonal = True
                m = 24  # 24 hours seasonality
            
            # Convert to numpy arrays if they're pandas objects
            if hasattr(y_train, 'values'):
                y_train_values = y_train.values
            else:
                y_train_values = y_train
            
            if hasattr(exog_train, 'values'):
                exog_train_values = exog_train.values
            else:
                exog_train_values = exog_train
                
            # Get statistics for scaling and validation
            train_mean = np.mean(y_train_values)
            train_std = np.std(y_train_values)
            train_min = np.min(y_train_values)
            train_max = np.max(y_train_values)
            
            # Define reasonable bounds based on training data
            lower_bound = max(0, train_min - 2 * train_std)
            upper_bound = train_max + 3 * train_std
            
            print(f"Training data statistics: mean={train_mean:.2f}, std={train_std:.2f}, min={train_min:.2f}, max={train_max:.2f}")
            print(f"Setting prediction bounds to: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Try to find optimal parameters with auto_arima - use simpler parameters to prevent overfitting
            try:
                print("Finding optimal SARIMAX parameters...")
                
                # Use auto_arima with exogenous variables - with more restricted parameters
                auto_model = auto_arima(
                    y_train_values,
                    exogenous=exog_train_values,
                    seasonal=seasonal,
                    m=m,
                    start_p=0, start_q=0,
                    max_p=1, max_q=1,  # Reduced to avoid overfitting
                    max_P=1, max_Q=1,  # Reduced to avoid overfitting
                    max_d=1, max_D=1,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                
                # Get the selected order and seasonal order
                order = auto_model.order
                seasonal_order = auto_model.seasonal_order if seasonal else (0, 0, 0, 0)
                print(f"Best SARIMAX order: {order}")
                print(f"Best SARIMAX seasonal order: {seasonal_order}")
                
            except Exception as e:
                print(f"Could not find optimal parameters: {e}")
                # Use very simple model as fallback
                order = (1, 0, 0)
                seasonal_order = (0, 0, 0, 0)  # No seasonality as fallback
                print(f"Using fallback simple SARIMAX order: {order}")
            
            # Train SARIMAX model with conservative settings
            model = SARIMAX(
                y_train_values,
                exog=exog_train_values,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,  # Changed to True
                enforce_invertibility=True,  # Changed to True
                simple_differencing=True    # Added to improve stability
            )
            
            # Fit the model with conservative settings
            print("Fitting SARIMAX model...")
            sarimax_fit = model.fit(disp=False, maxiter=50)
            
            # Convert test data to numpy if needed
            if hasattr(exog_test, 'values'):
                exog_test_values = exog_test.values
            else:
                exog_test_values = exog_test
                
            # Forecast
            print("Generating SARIMAX forecast...")
            y_pred = sarimax_fit.forecast(steps=len(y_test), exog=exog_test_values)
            
            # Convert to numpy array if it's a pandas Series
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            
            # Check for unreasonable predictions and clip them
            original_pred = y_pred.copy()
            y_pred = np.clip(y_pred, lower_bound, upper_bound)
            
            # Count how many predictions needed to be clipped
            num_clipped = np.sum(original_pred != y_pred)
            if num_clipped > 0:
                print(f"Warning: {num_clipped} out of {len(y_pred)} predictions were clipped to be within reasonable bounds")
            
            # Check if predictions are wildly off before proceeding
            pred_mean = np.mean(y_pred)
            pred_std = np.std(y_pred)
            
            # Skip model if the predictions are unreasonable
            if pred_mean > 5 * train_mean or pred_std > 5 * train_std:
                print(f"Warning: SARIMAX predictions are unreasonable (mean={pred_mean:.2f}, std={pred_std:.2f})")
                print("Skipping SARIMAX model in comparison to avoid distorting graphs")
                return None
            
            # Ensure predictions are non-negative
            y_pred = np.maximum(y_pred, 0)
            
            # Evaluate
            model_name = f"SARIMAX{order}"
            result = self.evaluate_model(y_test, y_pred, model_name)
            
            # Only proceed if results are reasonable
            if result['rmse'] > 5 * train_std:
                print(f"Warning: SARIMAX model has very high RMSE ({result['rmse']:.2f}) compared to data standard deviation ({train_std:.2f})")
                print("Skipping SARIMAX model in comparison to avoid distorting graphs")
                # Remove from model_results to avoid affecting the graphs
                if hasattr(self, 'model_results'):
                    self.model_results = [r for r in self.model_results if r['model'] != model_name]
                return None
            
            # Plot only if results are reasonable
            self.plot_predictions(y_test, y_pred, model_name, test_index)
            
            # Store the model
            self.models[model_name] = sarimax_fit
            
            return result
            
        except ImportError:
            print("statsmodels or pmdarima not installed. Skipping SARIMAX model.")
            return None
        except Exception as e:
            print(f"Error training SARIMAX model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_ml_models(self, X_train, X_test, y_train, y_test, test_index):
        """
        Train multiple machine learning models for time series forecasting.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target values
            y_test: Testing target values
            test_index: Index for test data
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Define models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        # Train each model and evaluate
        for name, model in models.items():
            try:
                print(f"\nTraining {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Ensure predictions are non-negative
                y_pred = np.maximum(y_pred, 0)
                
                # Evaluate
                result = self.evaluate_model(y_test, y_pred, name)
                results.append(result)
                
                # Plot
                self.plot_predictions(y_test, y_pred, name, test_index)
                
                # Store the model
                self.models[name] = model
                
                # Print feature importances for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = X_train.columns
                    
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    # Print top 10 features
                    print(f"\nTop 10 important features for {name}:")
                    for i in range(min(10, len(indices))):
                        print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        return results
    
    def train_and_evaluate_models(self, skip_sarimax=False):
        """
        Train and evaluate all models for air quality prediction.
        
        Args:
            skip_sarimax: If True, skip training the SARIMAX model
            
        Returns:
            The name of the best performing model
        """
        if self.forecast_data is None:
            print("No forecast data prepared. Call prepare_data_for_forecasting() first.")
            return
        
        # Split data into training and testing sets
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test, test_index = self.train_test_split(test_size=0.2)
        
        if X_train is None:
            print("Error splitting data. Check the train_test_split method.")
            return
        
        print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
        
        # Train machine learning models
        print("\n=== Training Machine Learning Models ===")
        self.train_ml_models(X_train, X_test, y_train, y_test, test_index)
        
        # # Train ARIMA model (univariate time series)
        # print("\n=== Training ARIMA Model ===")
        # self.train_arima_model(y_train, y_test, test_index)
        
        # # Train Prophet model (univariate time series with seasonality)
        # print("\n=== Training Prophet Model ===")
        # self.train_prophet_model(y_train, y_test, test_index)
        
        # # Train SARIMAX model (using exogenous variables) if not skipped
        # if not skip_sarimax and len(X_train.columns) > 0:
        #     print("\n=== Training SARIMAX Model ===")
        #     # Select top correlated features for exogenous variables
        #     exog_train = X_train.iloc[:, :min(5, X_train.shape[1])]  # Use top 5 features
        #     exog_test = X_test.iloc[:, :min(5, X_train.shape[1])]
        #     self.train_sarimax_model(y_train, exog_train, y_test, exog_test, test_index)
        # elif skip_sarimax:
        #     print("\n=== Skipping SARIMAX Model as requested ===")
        
       
        return
    
    def forecast_future(self, model_name, steps=24):
        """
        Generate forecasts for future time periods with uncertainty estimates.
        
        Args:
            model_name: Name of the model to use
            steps: Number of future steps to forecast
            
        Returns:
            DataFrame with forecasted values and confidence intervals
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Train models first.")
            return None
        
        model = self.models[model_name]
        
        # Create dates for future forecast
        last_date = self.forecast_data.index[-1] if self.forecast_data is not None else datetime.now()
        future_dates = [last_date + timedelta(hours=i+1) for i in range(steps)]
        
        # Initialize forecast DataFrame
        forecast_df = pd.DataFrame(index=future_dates)
        forecast_df.index.name = 'time'
        
        # Different forecasting methods based on model type
        if 'Prophet' in model_name:
            try:
                # For Prophet models
                future = pd.DataFrame({'ds': future_dates})
                prophet_forecast = model.predict(future)
                
                # Extract predictions and intervals
                forecast_df['forecasted_aqi'] = prophet_forecast['yhat'].values
                forecast_df['lower_bound'] = prophet_forecast['yhat_lower'].values
                forecast_df['upper_bound'] = prophet_forecast['yhat_upper'].values
                
                # Make sure predictions are positive
                forecast_df['forecasted_aqi'] = forecast_df['forecasted_aqi'].clip(lower=0)
                forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
                
                # Add Prophet components for interpretation
                for component in ['trend', 'yearly', 'weekly', 'daily', 'hourly']:
                    if component in prophet_forecast.columns:
                        forecast_df[f'component_{component}'] = prophet_forecast[component].values
            except Exception as e:
                print(f"Error forecasting with Prophet: {e}")
                return None
                
        elif 'ARIMA' in model_name or 'SARIMAX' in model_name:
            try:
                # For ARIMA and SARIMAX models
                forecast_result = model.get_forecast(steps=steps)
                
                # Extract predictions
                forecast_df['forecasted_aqi'] = forecast_result.predicted_mean
                
                # Add confidence intervals
                try:
                    # This might fail if confidence intervals cannot be computed
                    ci = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
                    forecast_df['lower_bound'] = ci.iloc[:, 0]
                    forecast_df['upper_bound'] = ci.iloc[:, 1]
                    
                    # Make sure bounds are positive
                    forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
                except Exception as e:
                    print(f"Could not compute confidence intervals: {e}")
                    # Generate simple intervals based on RMSE from model_results
                    if self.model_results:
                        for result in self.model_results:
                            if result['model'] == model_name and 'rmse' in result:
                                rmse = result['rmse']
                                forecast_df['lower_bound'] = forecast_df['forecasted_aqi'] - 1.96 * rmse
                                forecast_df['upper_bound'] = forecast_df['forecasted_aqi'] + 1.96 * rmse
                                forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
                                break
            except Exception as e:
                print(f"Error forecasting with {model_name}: {e}")
                return None
                
        elif 'VAR' in model_name or 'AR(' in model_name:
            try:
                # For VAR and AR models
                if hasattr(model, 'forecast'):
                    # VAR model
                    # Get the last observations from the data used to train
                    if hasattr(model, 'y') and model.y is not None:
                        last_obs = model.y[-model.k_ar:]
                        forecast_values = model.forecast(y=last_obs, steps=steps)
                        forecast_df['forecasted_aqi'] = forecast_values[:, 0]  # First column contains the target
                    else:
                        # Fallback if we can't access training data directly
                        print("Using AR forecast method for VAR model...")
                        forecast_df['forecasted_aqi'] = model.forecast(steps=steps)
                else:
                    # AR model
                    forecast_start = len(model.endog)
                    forecast_df['forecasted_aqi'] = model.predict(start=forecast_start, end=forecast_start + steps - 1)
                
                # Generate simple intervals based on RMSE from model_results
                if self.model_results:
                    for result in self.model_results:
                        if result['model'] == model_name and 'rmse' in result:
                            rmse = result['rmse']
                            forecast_df['lower_bound'] = forecast_df['forecasted_aqi'] - 1.96 * rmse
                            forecast_df['upper_bound'] = forecast_df['forecasted_aqi'] + 1.96 * rmse
                            forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
                            break
            except Exception as e:
                print(f"Error forecasting with {model_name}: {e}")
                return None
        else:
            # For ML models, we need to generate features for future time steps incrementally
            print(f"Generating features for future forecast using {model_name}...")
            
            # Get features needed for prediction
            if self.forecast_data is None:
                print("No forecast data available for generating features.")
                return None
            
            # Get the last few days of historical data to create lagged features
            max_lag = 168  # 1 week of hourly data
            history_needed = min(max_lag, len(self.forecast_data))
            recent_data = self.forecast_data.iloc[-history_needed:].copy()
            
            # Get the column we're forecasting
            target_col = 'calculated_aqi'
            
            # Create a DataFrame that will extend with each new forecast
            forecast_df = pd.DataFrame(index=future_dates)
            
            # Get the feature columns used during training
            if isinstance(model, LinearRegression) or isinstance(model, Ridge):
                if hasattr(model, 'feature_names_in_'):
                    feature_cols = model.feature_names_in_
                else:
                    # For older scikit-learn versions
                    # Get all columns except target from forecast_data
                    feature_cols = [col for col in self.forecast_data.columns if col != target_col]
            else:
                # For other model types
                feature_cols = [col for col in self.forecast_data.columns if col != target_col]
            
            # Dictionary to hold uncertainty estimates
            uncertainty = {
                'lower': [],
                'upper': []
            }
            
            # Generate predictions incrementally
            for i, future_date in enumerate(future_dates):
                # Create a new row with the forecasting date
                current_row = pd.Series(index=self.forecast_data.columns, name=future_date)
                
                # Add time features
                current_row['hour'] = future_date.hour
                current_row['day_of_week'] = future_date.dayofweek
                current_row['month'] = future_date.month
                current_row['day'] = future_date.day
                current_row['is_weekend'] = 1 if future_date.dayofweek >= 5 else 0
                
                # Add cyclical encoding for time
                current_row['hour_sin'] = np.sin(current_row['hour'] * (2 * np.pi / 24))
                current_row['hour_cos'] = np.cos(current_row['hour'] * (2 * np.pi / 24))
                current_row['day_of_week_sin'] = np.sin(current_row['day_of_week'] * (2 * np.pi / 7))
                current_row['day_of_week_cos'] = np.cos(current_row['day_of_week'] * (2 * np.pi / 7))
                current_row['month_sin'] = np.sin(current_row['month'] * (2 * np.pi / 12))
                current_row['month_cos'] = np.cos(current_row['month'] * (2 * np.pi / 12))
                
                # Add time period flags
                current_row['is_morning_rush'] = 1 if 7 <= future_date.hour <= 9 else 0
                current_row['is_evening_rush'] = 1 if 17 <= future_date.hour <= 19 else 0
                current_row['is_night'] = 1 if future_date.hour >= 22 or future_date.hour <= 5 else 0
                
                # Add most recent weather data (persistent forecast)
                weather_cols = ['temperature_c', 'humidity_pct', 'wind_speed_kmh', 'weather_category']
                for col in weather_cols:
                    if col in recent_data.columns:
                        current_row[col] = recent_data[col].iloc[-1]
                
                # Add location features if they exist
                location_cols = [col for col in recent_data.columns if col.startswith('location_')]
                for col in location_cols:
                    current_row[col] = recent_data[col].iloc[-1]
                
                # Add lagged features from past observations and predictions
                # Recent history + previous forecasts
                combined_history = pd.concat([recent_data, pd.DataFrame([current_row])])
                
                # Update lagged features
                for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
                    lag_col = f'{target_col}_lag_{lag}'
                    if lag_col in feature_cols:
                        if i < lag:
                            # Use value from history
                            idx = -lag + i
                            current_row[lag_col] = recent_data[target_col].iloc[idx] if idx < 0 else np.nan
                        else:
                            # Use previously forecasted value
                            current_row[lag_col] = forecast_df['forecasted_aqi'].iloc[i-lag]
                
                # Add rolling window features
                for window in [3, 6, 12, 24, 48, 72]:
                    window_mean_col = f'{target_col}_rolling_{window}h'
                    window_std_col = f'{target_col}_rolling_{window}h_std'
                    
                    if window_mean_col in feature_cols:
                        # Get last n forecasted values + required history
                        if i < window - 1:
                            # Need some history values
                            if len(recent_data) >= window - i - 1:
                                history_vals = recent_data[target_col].iloc[-(window-i-1):].tolist()
                            else:
                                history_vals = recent_data[target_col].tolist()
                                
                            # Combine with already forecasted values
                            if i > 0:
                                forecast_vals = forecast_df['forecasted_aqi'].iloc[:i].tolist()
                                all_vals = history_vals + forecast_vals
                            else:
                                all_vals = history_vals
                        else:
                            # Use only forecasted values
                            all_vals = forecast_df['forecasted_aqi'].iloc[i-window+1:i].tolist()
                        
                        # Compute mean and std
                        if all_vals:
                            current_row[window_mean_col] = np.mean(all_vals)
                            if window_std_col in feature_cols:
                                current_row[window_std_col] = np.std(all_vals) if len(all_vals) > 1 else 0
                
                # Add other statistical features
                if f'{target_col}_hour_avg' in feature_cols:
                    hour_values = recent_data[recent_data.index.hour == future_date.hour][target_col]
                    if len(hour_values) > 0:
                        current_row[f'{target_col}_hour_avg'] = hour_values.mean()
                
                if f'{target_col}_dow_avg' in feature_cols:
                    dow_values = recent_data[recent_data.index.dayofweek == future_date.dayofweek][target_col]
                    if len(dow_values) > 0:
                        current_row[f'{target_col}_dow_avg'] = dow_values.mean()
                
                # Clean up any NaN values
                for col in feature_cols:
                    if pd.isna(current_row[col]):
                        # Use column mean from training data
                        if col in self.forecast_data.columns:
                            current_row[col] = self.forecast_data[col].mean()
                        else:
                            current_row[col] = 0  # Default to 0 for missing columns
                
                # Create feature vector for prediction
                feature_vector = current_row[feature_cols].values.reshape(1, -1)
                
                # Scale the features if a scaler was used
                if self.scaler is not None:
                    try:
                        numeric_features = [col for col in feature_cols if col in self.scaler.feature_names_in_]
                        numeric_indices = [feature_cols.tolist().index(col) for col in numeric_features]
                        
                        if numeric_indices:
                            # Only scale the numeric features
                            feature_vector_df = pd.DataFrame(feature_vector, columns=feature_cols)
                            feature_vector_df.iloc[0, numeric_indices] = self.scaler.transform(
                                feature_vector_df.iloc[0:1, numeric_indices])[0]
                            feature_vector = feature_vector_df.values
                    except Exception as e:
                        print(f"Warning: Could not scale features: {e}")
                
                # Make the prediction
                try:
                    prediction = model.predict(feature_vector)[0]
                    
                    # Ensure prediction is non-negative
                    prediction = max(0, prediction)
                    
                    # Store the prediction
                    forecast_df.loc[future_date, 'forecasted_aqi'] = prediction
                    
                    # Update current_row for next iteration
                    current_row[target_col] = prediction
                    
                    # Generate uncertainty estimates
                    # 1. For tree-based models, get prediction intervals if available
                    if isinstance(model, RandomForestRegressor) or isinstance(model, GradientBoostingRegressor):
                        try:
                            if hasattr(model, 'estimators_'):
                                # For Random Forest, use predictions from all trees
                                predictions = np.array([tree.predict(feature_vector)[0] for tree in model.estimators_])
                                lower = np.percentile(predictions, 5)
                                upper = np.percentile(predictions, 95)
                                uncertainty['lower'].append(max(0, lower))
                                uncertainty['upper'].append(upper)
                            else:
                                # Use RMSE from model results
                                if self.model_results:
                                    for result in self.model_results:
                                        if result['model'] == model_name and 'rmse' in result:
                                            rmse = result['rmse']
                                            uncertainty['lower'].append(max(0, prediction - 1.96 * rmse))
                                            uncertainty['upper'].append(prediction + 1.96 * rmse)
                                            break
                        except Exception:
                            # Fallback to RMSE-based intervals
                            if self.model_results:
                                for result in self.model_results:
                                    if result['model'] == model_name and 'rmse' in result:
                                        rmse = result['rmse']
                                        uncertainty['lower'].append(max(0, prediction - 1.96 * rmse))
                                        uncertainty['upper'].append(prediction + 1.96 * rmse)
                                        break
                    else:
                        # For other models, use RMSE from model results
                        if self.model_results:
                            for result in self.model_results:
                                if result['model'] == model_name and 'rmse' in result:
                                    rmse = result['rmse']
                                    uncertainty['lower'].append(max(0, prediction - 1.96 * rmse))
                                    uncertainty['upper'].append(prediction + 1.96 * rmse)
                                    break
                
                except Exception as e:
                    print(f"Error predicting for {future_date}: {str(e)}")
                    # Fall back to using the last known value
                    last_value = recent_data[target_col].iloc[-1]
                    forecast_df.loc[future_date, 'forecasted_aqi'] = last_value
                    
                    # Add uncertainty based on model results
                    if self.model_results:
                        for result in self.model_results:
                            if result['model'] == model_name and 'rmse' in result:
                                rmse = result['rmse']
                                uncertainty['lower'].append(max(0, last_value - 1.96 * rmse))
                                uncertainty['upper'].append(last_value + 1.96 * rmse)
                                break
            
            # Add uncertainty intervals to forecast if calculated
            if uncertainty['lower'] and uncertainty['upper']:
                forecast_df['lower_bound'] = uncertainty['lower']
                forecast_df['upper_bound'] = uncertainty['upper']
            
            return forecast_df
        
        return forecast_df
        
    def save_models(self, output_dir='models'):
        """
        Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models to
            
        Returns:
            Boolean indicating if saving was successful
        """
        if not self.models:
            print("No models to save. Train models first.")
            return False
            
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each model
            saved_models = []
            
            for model_name, model in self.models.items():
                try:
                    # Create a clean filename
                    clean_name = ''.join(c if c.isalnum() else '_' for c in model_name)
                    model_path = os.path.join(output_dir, f"{clean_name}.pkl")
                    
                    # Save the model
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                        
                    saved_models.append(model_name)
                    print(f"Saved model '{model_name}' to {model_path}")
                    
                except Exception as e:
                    print(f"Error saving model '{model_name}': {e}")
            
            # Save metadata
            metadata = {
                'saved_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'models': saved_models,
                'model_metrics': self.model_results if hasattr(self, 'model_results') else None
            }
            
            metadata_path = os.path.join(output_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
            print(f"Saved model metadata to {metadata_path}")
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
            
    def load_models(self, input_dir='models'):
        """
        Load saved models from disk.
        
        Args:
            input_dir: Directory to load models from
            
        Returns:
            Boolean indicating if loading was successful
        """
        if not os.path.exists(input_dir):
            print(f"Input directory '{input_dir}' does not exist.")
            return False
            
        try:
            # Count models loaded
            loaded_count = 0
            
            # Get all pickle files in the directory
            model_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
            
            if not model_files:
                print(f"No model files found in '{input_dir}'.")
                return False
                
            # Load each model
            for model_file in model_files:
                try:
                    model_path = os.path.join(input_dir, model_file)
                    model_name = os.path.splitext(model_file)[0].replace('_', ' ')
                    
                    # Load the model
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        
                    # Store the model
                    self.models[model_name] = model
                    loaded_count += 1
                    print(f"Loaded model '{model_name}' from {model_path}")
                    
                except Exception as e:
                    print(f"Error loading model '{model_file}': {e}")
            
            # Try to load model metrics
            metadata_path = os.path.join(input_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    if 'model_metrics' in metadata and metadata['model_metrics']:
                        self.model_results = metadata['model_metrics']
                        print(f"Loaded model metrics from {metadata_path}")
                except Exception as e:
                    print(f"Error loading model metadata: {e}")
            
            print(f"Successfully loaded {loaded_count} models.")
            return loaded_count > 0
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

