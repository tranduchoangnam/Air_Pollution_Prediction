#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
from datetime import datetime

# Add the parent directory to the path for importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.timescaledb_util import TimescaleDBUtil

def log(message):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {message}")

def create_merge_aqi_weather_table():
    """
    Merge data from mttd_aqi and airvisual_waqi_merge_aqi tables into a new merge_aqi_weather table.
    
    The resulting table contains:
    - time, location (unique key)
    - Pollutant data: PM2.5, PM10, NO2, SO2, O3, CO
    - Weather data: temperature, pressure, humidity, wind_speed, wind_direction, weather_icon, dew
    
    Only processes records newer than the latest timestamp in the existing merge_aqi_weather table.
    """
    log("Starting merge process...")
    
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
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
        
        if 'weather_icon' in airvisual_hourly.columns:
            # Fill missing weather_icon values with the most common value
            most_common_icon = airvisual_hourly['weather_icon'].mode().iloc[0] if not airvisual_hourly['weather_icon'].mode().empty else '01d'
            airvisual_hourly['weather_icon'].fillna(most_common_icon, inplace=True)
        
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
                weather_data[col] = weather_data[col].fillna(method='ffill').fillna(method='bfill')
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
                        lambda x: x.fillna(method='ffill').fillna(method='bfill')
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
    success = create_merge_aqi_weather_table()
    
    if success:
        log("✅ Merge process completed successfully")
    else:
        log("❌ Merge process failed")
    
    log("=== Merge Process Completed ===")

if __name__ == "__main__":
    main() 