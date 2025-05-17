#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import pytz
from datetime import datetime

# Add the parent directory to the path for importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.timescaledb_util import TimescaleDBUtil
from moitruongthudo.air_quality_prediction import AirQualityPrediction

def log(message):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {message}")

def create_merge_aqi_weather_table():
    """
    Merge data from mttd_aqi and airvisual_waqi_merge_aqi tables into a new merge_aqi_weather table.
    Also incorporates pollutant data from air_quality table when available.
    
    The resulting table contains:
    - time, location (unique key)
    - Pollutant data: PM2.5, PM10, NO2, SO2, O3, CO
    - Weather data: temperature, pressure, humidity, wind_speed, wind_direction, weather_icon, dew
    
    Only processes records newer than the latest timestamp in the existing merge_aqi_weather table.
    """
    log("Starting merge process...")
    
    # Define location name mapping for air_quality table
    location_mapping = {
        'ao Duy Tu': 'Đào Duy Từ',
        'An Khanh': 'An Khánh',
        'Minh Khai Bac Tu Liem': 'Minh Khai - Bắc Từ Liêm',
        'So 46 pho Luu Quang Vu': '46 Lưu Quang Vũ'
    }
    
    # Connect to TimescaleDB
    ts_db = TimescaleDBUtil()
    if not ts_db.connect():
        log("[!] Failed to connect to TimescaleDB")
        return False
    
    try:
        # Check if the merge_aqi_weather table exists
        check_table_query = "SELECT to_regclass('public.merge_aqi_weather')"
        result = ts_db.execute_query(check_table_query)
        table_exists = result is not None and result.iloc[0, 0] is not None
        
        latest_timestamp = None
        if table_exists:
            # Get the latest timestamp from the existing table
            latest_timestamp_query = "SELECT MAX(time) FROM merge_aqi_weather"
            latest_result = ts_db.execute_query(latest_timestamp_query)
            if latest_result is not None and not latest_result.empty and latest_result.iloc[0, 0] is not None:
                latest_timestamp = latest_result.iloc[0, 0]
                log(f"Found existing merge_aqi_weather table with latest timestamp: {latest_timestamp}")
                
                # Include a buffer to avoid missing records due to slight timing differences
                # Subtract 1 hour from latest timestamp to ensure overlap
                buffer_time = pd.to_datetime(latest_timestamp) - pd.Timedelta(hours=1)
                log(f"Using buffer time of 1 hour: fetching records from {buffer_time} onwards")
                latest_timestamp = buffer_time
            else:
                log("Found existing merge_aqi_weather table but couldn't determine latest timestamp")
        else:
            log("Table merge_aqi_weather does not exist. Will create a new table.")
            
        # Only process new records if we have determined a latest timestamp
        time_filter = ""
        if latest_timestamp:
            time_filter = f"WHERE time > '{latest_timestamp}'"
            log(f"Will only process records newer than {latest_timestamp}")
            
        # Step 1: Get data from mttd_aqi table
        log("Fetching data from mttd_aqi table...")
        mttd_query = f"""
            SELECT 
                time,
                station as location,
                "PM2_5_AQI" as "PM2.5",
                "PM10_AQI" as "PM10",
                "O3_AQI" as "O3",
                "SO2_AQI" as "SO2",
                "NO2_AQI" as "NO2",
                "CO_AQI" as "CO"
            FROM mttd_aqi
            {time_filter}
        """
        mttd_df = ts_db.execute_query(mttd_query)
        if mttd_df is None or mttd_df.empty:
            log("[!] No new data found in mttd_aqi table")
            return True  # Return success as no new data is not an error
        
        log(f"Retrieved {len(mttd_df)} new records from mttd_aqi")
        
        # Step 2: Get data from airvisual_waqi_merge_aqi table - need to get data from a wider timespan
        # to ensure we have enough weather data for interpolation
        timespan_buffer = ""
        if latest_timestamp:
            # Get data from 24 hours before the latest timestamp to ensure smooth interpolation
            buffer_timestamp = pd.to_datetime(latest_timestamp) - pd.Timedelta(hours=24)
            timespan_buffer = f"WHERE timestamp > '{buffer_timestamp}'"
            log(f"Fetching weather data from {buffer_timestamp} onwards (24-hour buffer)")
        else:
            log("Fetching all available weather data")
            
        # Step 2.5: Get data from air_quality table for additional pollutant data
        air_quality_time_filter = ""
        if latest_timestamp:
            # Convert to UTC for comparison with air_quality datetimeLocal
            # Latest timestamp is in UTC+7 without timezone info, so first make it timezone-aware
            latest_timestamp_aware = pd.to_datetime(latest_timestamp).tz_localize(pytz.timezone('Asia/Bangkok'))
            # Then convert to UTC
            latest_timestamp_utc = latest_timestamp_aware.astimezone(pytz.UTC)
            # Apply buffer in UTC
            buffer_timestamp_utc = latest_timestamp_utc - pd.Timedelta(hours=1)
            air_quality_time_filter = f"WHERE \"datetimeLocal\" > '{buffer_timestamp_utc}'"
            log(f"Fetching air_quality data from {buffer_timestamp_utc} onwards (UTC time)")
        else:
            log("Fetching all available air_quality data")
            
        log("Fetching data from air_quality table...")
        air_quality_query = f"""
            SELECT 
                "datetimeLocal" as utc_time,
                location_name,
                "pm2.5",
                "pm10",
                "o3",
                "so2",
                "no2",
                "co"
            FROM air_quality
            {air_quality_time_filter}
        """
        air_quality_df = ts_db.execute_query(air_quality_query)
        if air_quality_df is None or air_quality_df.empty:
            log("[!] No data found in air_quality table for the specified timespan")
            air_quality_df = pd.DataFrame(columns=['utc_time', 'location_name', 'pm2.5', 'pm10', 'o3', 'so2', 'no2', 'co'])
        else:
            log(f"Retrieved {len(air_quality_df)} records from air_quality table")
            
            # Initialize AQI calculator
            aqi_calculator = AirQualityPrediction()
            
            # Process air_quality data
            # 1. Convert UTC time to UTC+7 and remove timezone info
            air_quality_df['utc_time'] = pd.to_datetime(air_quality_df['utc_time'])
            # Check if timestamps are already timezone-aware
            if air_quality_df['utc_time'].dt.tz is None:
                air_quality_df['time'] = air_quality_df['utc_time'].dt.tz_localize(pytz.UTC).dt.tz_convert(pytz.timezone('Asia/Bangkok')).dt.tz_localize(None)
            else:
                air_quality_df['time'] = air_quality_df['utc_time'].dt.tz_convert(pytz.timezone('Asia/Bangkok')).dt.tz_localize(None)
            
            # 2. Map location names
            air_quality_df['location'] = air_quality_df['location_name'].map(location_mapping).fillna(air_quality_df['location_name'])
            
            # 3. Rename pollutant columns to match our naming convention and round to hourly timestamps
            air_quality_df.rename(columns={'pm2.5': 'PM2.5', 'pm10': 'PM10', 'o3': 'O3', 'so2': 'SO2', 'no2': 'NO2', 'co': 'CO'}, inplace=True)
            air_quality_df['time'] = air_quality_df['time'].dt.floor('h')
            
            # 4. Convert values to numeric and aggregate to hourly resolution
            for col in ['PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO']:
                air_quality_df[col] = pd.to_numeric(air_quality_df[col], errors='coerce')
            
            # 5. Calculate AQI values for each pollutant
            log("Calculating AQI values for pollutants from air_quality table...")
            aqi_mapping = {
                'PM2.5': 'pm25',     # Mapping for AirQualityPrediction.calculate_aqi
                'PM10': 'pm10',      # These match the parameter names expected by calculate_aqi
                'O3': 'o3',
                'SO2': 'so2',
                'NO2': 'no2',
                'CO': 'co'
            }
            
            # Create columns for AQI values
            for col, aqi_param in aqi_mapping.items():
                if col in air_quality_df.columns:
                    # Only process rows with non-null values
                    # air_quality_df[f'{col}_AQI'] = None
                    mask = ~air_quality_df[col].isna()
                    if mask.any():
                        # Apply AQI calculation to each value
                        air_quality_df.loc[mask, col] = air_quality_df.loc[mask, col].apply(
                            lambda x: aqi_calculator.calculate_aqi(x, aqi_param) if pd.notna(x) else None
                        )
            
            # Aggregate by hour and location
            agg_dict = {col: 'mean' for col in ['PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO']}
                
            air_quality_hourly = air_quality_df.groupby(['time', 'location']).agg(agg_dict).reset_index()
            
            log(f"Processed {len(air_quality_hourly)} hourly records from air_quality table with AQI calculations")
        
        log("Fetching data from airvisual_waqi_merge_aqi table...")
        airvisual_query = f"""
            SELECT 
                timestamp as time,
                COALESCE(city, 'Hai BaTrung') as location,
                "PM2.5" as "PM2.5",
                "PM10" as "PM10",
                "O3" as "O3",
                "SO2" as "SO2",
                "NO2" as "NO2",
                "CO" as "CO",
                temperature,
                pressure,
                humidity,
                wind_speed,
                wind_direction,
                weather_icon,
                dew
            FROM airvisual_waqi_merge_aqi
            {timespan_buffer}
        """
        airvisual_df = ts_db.execute_query(airvisual_query)
        if airvisual_df is None or airvisual_df.empty:
            log("[!] No data found in airvisual_waqi_merge_aqi table for the specified timespan")
            return True  # Return success as no new data is not an error
        
        log(f"Retrieved {len(airvisual_df)} records from airvisual_waqi_merge_aqi")
        
        # Step 3: Merge the dataframes
        log("Merging datasets...")
        
        # Ensure time columns are in datetime format
        mttd_df['time'] = pd.to_datetime(mttd_df['time'])
        airvisual_df['time'] = pd.to_datetime(airvisual_df['time'])
        
        # Round timestamps to the nearest hour for consistent merging
        mttd_df['time'] = mttd_df['time'].dt.floor('h')  # Using 'h' instead of 'H' to avoid deprecation warning
        
        # Create a copy of airvisual_df with original timestamps before rounding
        log("Aggregating airvisual_waqi_merge_aqi data to hourly resolution...")
        airvisual_hourly = airvisual_df.copy()
        airvisual_hourly['hour'] = airvisual_hourly['time'].dt.floor('h')
        
        # Convert columns to numeric, errors='coerce' will convert non-numeric values to NaN
        numeric_columns = ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction', 'dew', 
                          'PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO']
        
        for col in numeric_columns:
            if col in airvisual_hourly.columns:
                airvisual_hourly[col] = pd.to_numeric(airvisual_hourly[col], errors='coerce')
        
        # Fill missing weather values before aggregation
        weather_columns = ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction', 'dew']
        for col in weather_columns:
            if col in airvisual_hourly.columns:
                # Forward fill within each location group
                airvisual_hourly[col] = airvisual_hourly.groupby('location')[col].transform(
                    lambda x: x.ffill().bfill()
                )
        
        if 'weather_icon' in airvisual_hourly.columns:
            # Fill missing weather_icon values with the most common value
            most_common_icon = airvisual_hourly['weather_icon'].mode().iloc[0] if not airvisual_hourly['weather_icon'].mode().empty else '01d'
            airvisual_hourly['weather_icon'] = airvisual_hourly['weather_icon'].fillna(most_common_icon)
        
        # Group by hour and location, then aggregate
        agg_dict = {col: 'mean' for col in numeric_columns if col in airvisual_hourly.columns}
        # For weather_icon, take the most common value in each hour
        if 'weather_icon' in airvisual_hourly.columns:
            agg_dict['weather_icon'] = lambda x: x.mode().iloc[0] if not x.mode().empty else most_common_icon
        
        # Perform the aggregation
        airvisual_hourly = airvisual_hourly.groupby(['hour', 'location']).agg(agg_dict).reset_index()
        airvisual_hourly.rename(columns={'hour': 'time'}, inplace=True)
        
        log(f"Aggregated to {len(airvisual_hourly)} hourly records")
        
        # Extract weather data from airvisual_hourly (all from Hai BaTrung)
        log("Extracting weather data to apply to all locations...")
        weather_columns = ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction', 'weather_icon', 'dew']
        hai_batrung_weather = airvisual_hourly[airvisual_hourly['location'] == 'Hai BaTrung'][['time'] + weather_columns].copy()
        
        # Create a continuous time series for all hours in our dataset
        log("Creating continuous hourly weather data...")
        min_time = min(mttd_df['time'].min(), hai_batrung_weather['time'].min())
        max_time = max(mttd_df['time'].max(), hai_batrung_weather['time'].max())
        
        # Generate a complete range of hourly timestamps
        all_hours = pd.date_range(start=min_time, end=max_time, freq='h')
        complete_weather = pd.DataFrame({'time': all_hours})
        
        # Merge with existing weather data
        weather_data = pd.merge(
            complete_weather,
            hai_batrung_weather,
            on='time',
            how='left'
        )
        
        # Fill missing values using interpolation for numeric columns
        for col in weather_columns:
            if col == 'weather_icon':
                # Forward fill then backward fill for categorical data
                weather_data[col] = weather_data[col].ffill().bfill()
            else:
                # Use interpolation for numeric columns
                weather_data[col] = pd.to_numeric(weather_data[col], errors='coerce')
                # Linear interpolation between known points
                weather_data[col] = weather_data[col].interpolate(method='linear', limit_direction='both')
        
        # Ensure we have a complete dataset of weather values
        for col in weather_columns:
            missing_count = weather_data[col].isna().sum()
            if missing_count > 0:
                log(f"Warning: Found {missing_count} missing values in {col} after interpolation. Filling with nearby values.")
                if col == 'weather_icon':
                    # For categorical, use mode
                    most_common = weather_data[col].mode()[0]
                    weather_data[col] = weather_data[col].fillna(most_common)
                else:
                    # For numeric, use mean
                    col_mean = weather_data[col].mean()
                    weather_data[col] = weather_data[col].fillna(col_mean)
        
        log(f"Created continuous weather data for {len(weather_data)} hours")
        
        # Merge weather data with pollutant data based on timestamp only
        log("Applying weather data to all locations with the same timestamp...")
        merged_df = pd.merge(
            mttd_df,
            weather_data,
            on='time',
            how='left'
        )
        
        # Keep all Hai BaTrung records (which now include pollutant data)
        hai_batrung_records = airvisual_hourly[airvisual_hourly['location'] == 'Hai BaTrung'].copy()
        
        # Filter to keep only records newer than the latest timestamp, if applicable
        if latest_timestamp:
            hai_batrung_records = hai_batrung_records[hai_batrung_records['time'] > pd.to_datetime(latest_timestamp)]
            log(f"Filtered to {len(hai_batrung_records)} new Hai BaTrung records")
        
        # Combine with merged data
        merged_df = pd.concat([merged_df, hai_batrung_records], ignore_index=True)
        
        # Merge in the air_quality data (only for PM2.5 and PM10)
        if not air_quality_df.empty and 'air_quality_hourly' in locals():
            log("Merging additional pollutant data from air_quality table...")
            
            # Create a copy of merged_df to avoid SettingWithCopyWarning
            merged_df_with_air_quality = merged_df.copy()
            
            # Count records before merging
            pollutant_columns = ['PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO']
            missing_before = {col: merged_df_with_air_quality[col].isna().sum() for col in pollutant_columns}
            
            # Create a composite key for merging
            merged_df_with_air_quality['merge_key'] = merged_df_with_air_quality['time'].astype(str) + '_' + merged_df_with_air_quality['location']
            air_quality_hourly['merge_key'] = air_quality_hourly['time'].astype(str) + '_' + air_quality_hourly['location']
            
            # For each location and time combination in air_quality_hourly
            updated_records = 0
            better_quality_records = 0
            for idx, row in air_quality_hourly.iterrows():
                merge_key = row['merge_key']
                match_mask = merged_df_with_air_quality['merge_key'] == merge_key
                
                if match_mask.any():
                    # Get the index of the matching record
                    match_idx = merged_df_with_air_quality.index[match_mask]
                    
                    # Count null values in both records to determine which has better quality
                    null_count_air_quality = 0
                    null_count_existing = 0
                    
                    # Examine both raw pollutant values and calculated AQI values
                    pollutant_columns = ['PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO']
                    
                    # Count nulls for pollutants in both datasets
                    for col in pollutant_columns:
                        if col in row and pd.isna(row[col]):
                            null_count_air_quality += 1
                        
                        if pd.isna(merged_df_with_air_quality.loc[match_idx, col].values[0]):
                            null_count_existing += 1
                    
                    # Determine which record has better quality (fewer nulls)
                    # Prefer air_quality if it has calculated AQI values or if quality is similar
                    use_air_quality = (null_count_air_quality <= null_count_existing)
                    
                    if use_air_quality:
                        better_quality_records += 1
                        # Use air_quality record as it has better quality
                        # Only copy over the basic pollutant values that are in the DB schema
                        for col in pollutant_columns:
                            if col in row and pd.notna(row[col]):
                                merged_df_with_air_quality.loc[match_idx, col] = row[col]
                                updated_records += 1
                    else:
                        # Keep existing record as it has better quality
                        # But still fill in any missing pollutant values from air_quality
                        for col in pollutant_columns:
                            if col in row and pd.notna(row[col]) and pd.isna(merged_df_with_air_quality.loc[match_idx, col].values[0]):
                                merged_df_with_air_quality.loc[match_idx, col] = row[col]
                                updated_records += 1
            
            # Remove the temporary merge key
            merged_df_with_air_quality.drop('merge_key', axis=1, inplace=True)
            
            # Count records after merging
            missing_after = {col: merged_df_with_air_quality[col].isna().sum() for col in pollutant_columns}
            
            # Log the number of filled values for each pollutant
            for col in pollutant_columns:
                filled = missing_before[col] - missing_after[col]
                if filled > 0:
                    log(f"Filled {filled} missing {col} values")
            
            log(f"Updated a total of {updated_records} pollutant values using air_quality data")
            log(f"Used air_quality as primary source for {better_quality_records} records based on AQI-calculated data quality")
            
            # Replace the original merged_df with the updated one
            merged_df = merged_df_with_air_quality
        
        # Remove duplicates based on time and location
        log("Handling potential duplicate records...")
        # Sort by time (descending) and location to keep the most recent records first
        merged_df = merged_df.sort_values(by=['time', 'location'], ascending=[False, True])
        
        # Count potential duplicates
        duplicate_count = merged_df.duplicated(subset=['time', 'location'], keep=False).sum()
        if duplicate_count > 0:
            log(f"Found {duplicate_count} potential duplicate records due to the time buffer overlap")
            
            # Strategy: For duplicates, keep the record with more non-null values
            # First identify duplicated rows
            is_duplicate = merged_df.duplicated(subset=['time', 'location'], keep=False)
            duplicated_df = merged_df[is_duplicate].copy()
            
            # For each duplicated (time, location) pair, keep the record with the most non-null values
            if len(duplicated_df) > 0:
                # Count non-null values for each record
                duplicated_df['non_null_count'] = duplicated_df.notna().sum(axis=1)
                
                # Get the index of the record with the most non-null values for each (time, location) group
                idx_to_keep = duplicated_df.groupby(['time', 'location'])['non_null_count'].idxmax().values
                
                # Create a mask for records to keep
                keep_mask = merged_df.index.isin(idx_to_keep) | ~is_duplicate
                
                # Apply the mask
                merged_df = merged_df[keep_mask].copy()
                
                log(f"After duplicate resolution, keeping {len(merged_df)} records with the best data quality")
            
            # Final check for any remaining duplicates
            final_duplicate_count = merged_df.duplicated(subset=['time', 'location'], keep=False).sum()
            if final_duplicate_count > 0:
                log(f"Still found {final_duplicate_count} duplicates after resolution, keeping first occurrence")
                merged_df = merged_df.drop_duplicates(subset=['time', 'location'], keep='first')
        else:
            log("No duplicate records found")
        
        # Verify no null weather values for Hai BaTrung records
        hai_batrung_merged = merged_df[merged_df['location'] == 'Hai BaTrung']
        weather_nulls = hai_batrung_merged[weather_columns].isna().sum().sum()
        if weather_nulls > 0:
            log(f"Warning: Found {weather_nulls} null weather values in 'Hai BaTrung' records after merging. Fixing these...")
            
            # Identify records with null weather values
            for col in weather_columns:
                null_count = hai_batrung_merged[col].isna().sum()
                if null_count > 0:
                    log(f"  - {null_count} null values in '{col}'")
                    
                    # Fill null values for this column across all records
                    merged_df[col] = merged_df.groupby('location')[col].transform(
                        lambda x: x.ffill().bfill()
                    )
        
        log(f"Merged dataset has {len(merged_df)} records after removing duplicates")
        
        if merged_df.empty:
            log("No new records to insert after processing")
            return True
            
        # Step 4: Create or append to the merge_aqi_weather table
        if not table_exists:
            # Create a new table if it doesn't exist
            log("Creating merge_aqi_weather table...")
            
            # Create the table
            create_table_query = """
            CREATE TABLE merge_aqi_weather (
                time TIMESTAMP NOT NULL,
                location TEXT NOT NULL,
                "PM2.5" DOUBLE PRECISION,
                "PM10" DOUBLE PRECISION,
                "NO2" DOUBLE PRECISION,
                "SO2" DOUBLE PRECISION,
                "O3" DOUBLE PRECISION,
                "CO" DOUBLE PRECISION,
                temperature DOUBLE PRECISION,
                pressure DOUBLE PRECISION,
                humidity DOUBLE PRECISION,
                wind_speed DOUBLE PRECISION,
                wind_direction DOUBLE PRECISION,
                weather_icon TEXT,
                dew DOUBLE PRECISION,
                PRIMARY KEY (time, location)
            )
            """
            ts_db.execute_query(create_table_query)
            
            # Convert the merged dataframe to a TimescaleDB hypertable
            log("Converting to TimescaleDB hypertable...")
            ts_db.create_hypertable('merge_aqi_weather', 'time')
            
            # Create indexes for better query performance
            log("Creating indexes...")
            ts_db.execute_query("CREATE INDEX IF NOT EXISTS idx_merge_aqi_weather_location ON merge_aqi_weather (location)")
            ts_db.execute_query("CREATE INDEX IF NOT EXISTS idx_merge_aqi_weather_time ON merge_aqi_weather (time)")
        else:
            log("Appending to existing merge_aqi_weather table...")
        
        # Step 5: Insert the merged data
        log("Inserting merged data into merge_aqi_weather table...")
        
        # Ensure column names match the table structure
        merged_df = merged_df.rename(columns={
            'PM2.5': 'PM2.5',
            'PM10': 'PM10',
            'NO2': 'NO2',
            'SO2': 'SO2',
            'O3': 'O3',
            'CO': 'CO'
        })
        
        # Insert data in batches
        batch_size = 1000
        total_records = len(merged_df)
        records_inserted = 0
        
        for i in range(0, total_records, batch_size):
            batch_df = merged_df.iloc[i:i+batch_size]
            
            # Use the flexible insert method with "on conflict do nothing" to handle any potential duplicates
            on_conflict_clause = """
                ON CONFLICT (time, location) 
                DO UPDATE SET
                    "PM2.5" = COALESCE(EXCLUDED."PM2.5", merge_aqi_weather."PM2.5"),
                    "PM10" = COALESCE(EXCLUDED."PM10", merge_aqi_weather."PM10"),
                    "NO2" = COALESCE(EXCLUDED."NO2", merge_aqi_weather."NO2"),
                    "SO2" = COALESCE(EXCLUDED."SO2", merge_aqi_weather."SO2"),
                    "O3" = COALESCE(EXCLUDED."O3", merge_aqi_weather."O3"),
                    "CO" = COALESCE(EXCLUDED."CO", merge_aqi_weather."CO"),
                    temperature = COALESCE(EXCLUDED.temperature, merge_aqi_weather.temperature),
                    pressure = COALESCE(EXCLUDED.pressure, merge_aqi_weather.pressure),
                    humidity = COALESCE(EXCLUDED.humidity, merge_aqi_weather.humidity),
                    wind_speed = COALESCE(EXCLUDED.wind_speed, merge_aqi_weather.wind_speed),
                    wind_direction = COALESCE(EXCLUDED.wind_direction, merge_aqi_weather.wind_direction),
                    weather_icon = COALESCE(EXCLUDED.weather_icon, merge_aqi_weather.weather_icon),
                    dew = COALESCE(EXCLUDED.dew, merge_aqi_weather.dew)
            """
            
            success = ts_db.flexible_insert_dataframe(
                df=batch_df,
                table_name='merge_aqi_weather',
                if_exists='append',
                batch_size=batch_size,
                on_conflict=on_conflict_clause
            )
            
            if not success:
                log(f"[!] Failed to insert batch starting at record {i}")
                continue
            
            records_inserted += len(batch_df)
            log(f"Progress: {records_inserted}/{total_records} records inserted ({records_inserted/total_records*100:.1f}%)")
        
        log(f"Successfully inserted {records_inserted} new records into merge_aqi_weather table")
        
        return True
        
    except Exception as e:
        log(f"[!] Error during merge process: {str(e)}")
        return False
    finally:
        ts_db.disconnect()

def main():
    """Main function to run the merge process"""
    log("=== Starting AQI and Weather Data Merge Process ===")
    
    # Check for --force flag to process all data
    force_all = "--force" in sys.argv
    if force_all:
        log("⚠️ Force flag detected - will process all data from scratch")
        # Drop the existing table if it exists
        ts_db = TimescaleDBUtil()
        if ts_db.connect():
            ts_db.execute_query("DROP TABLE IF EXISTS merge_aqi_weather")
            ts_db.disconnect()
            log("✅ Dropped existing merge_aqi_weather table")
    
    success = create_merge_aqi_weather_table()
    
    if success:
        log("✅ Merge process completed successfully")
    else:
        log("❌ Merge process failed")
    
    log("=== Merge Process Completed ===")

if __name__ == "__main__":
    main() 