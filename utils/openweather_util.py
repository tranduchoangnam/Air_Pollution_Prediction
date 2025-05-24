import requests
import pandas as pd
from datetime import datetime, timedelta
import os
API_KEY = '06d966b9180a7fe9559c93f6f544e3ad'
CSV_FILE = 'weather_forecast_data.csv'

# Danh sách vị trí
locations = {
    "An Khánh": (21.0024, 105.7181),
    "46 Lưu Quang Vũ": (21.0152, 105.7999),
    "Minh Khai - Bắc Từ Liêm": (21.05, 105.74),
    "Đào Duy Từ": (21.0354, 105.8529),
    "Vân Hà": (21.2183, 106.0376),
    "Hai BaTrung": (21.0041, 105.8555)
}

def convert_hourly_data(hourly_data):
    converted_data = []

    for hour in hourly_data:
        dt_utc = datetime.utcfromtimestamp(hour["dt"])
        dt_utc_plus7 = dt_utc + timedelta(hours=7)
        time_str = dt_utc_plus7.strftime("%Y-%m-%d %H:%M:%S")

        temp_celsius = round(hour["temp"] - 273.15, 2)
        dew_celsius = round(hour["dew_point"] - 273.15, 2)

        entry = {
            "time": time_str,
            "temperature": temp_celsius,
            "pressure": hour["pressure"],
            "humidity": hour["humidity"],
            "wind_speed": hour["wind_speed"],
            "wind_direction": hour["wind_deg"],
            "weather_icon": hour["weather"][0]["icon"],
            "dew": dew_celsius
        }

        converted_data.append(entry)

    return converted_data

def get_weather_forecast_next_24h(df, location_name):
    """
    Trả về DataFrame dự báo 24h tiếp theo tại vị trí cụ thể từ file CSV, merged với df đầu vào.
    """
    if not os.path.exists(CSV_FILE):
        raise Exception(f"CSV file {CSV_FILE} does not exist")

    # Read CSV
    csv_df = pd.read_csv(CSV_FILE)
    csv_df['time'] = pd.to_datetime(csv_df['time'])
    
    # Filter CSV data for the specified location and next 24 hours
    forecast_df = csv_df[
        (csv_df['location'] == location_name) 
    ]
    
    # Perform left merge with input df
    forecast_df = pd.merge(
        df,
        forecast_df,
        on=['time', 'location'],
        how='left'
    )
    
    return forecast_df

def fetch_weather_forecast(location_name, lat, lon, api_key=API_KEY):
    """
    Fetch 48-hour weather forecast for a specific location and return as DataFrame.
    """
    url = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,daily,current,alerts&appid={api_key}'
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Request failed for {location_name} with status code {response.status_code}")

    data = response.json()
    hourly_data = data.get("hourly", [])
    forecast_list = convert_hourly_data(hourly_data)
    forecast_df = pd.DataFrame(forecast_list)
    forecast_df['time'] = pd.to_datetime(forecast_df['time'])
    forecast_df['location'] = location_name
    return forecast_df

if __name__ == "__main__":
    # Initialize an empty list to collect new data
    all_forecasts = []
    
    # Fetch forecasts for all locations
    for location_name, (lat, lon) in locations.items():
        try:
            forecast_df = fetch_weather_forecast(location_name, lat, lon)
            all_forecasts.append(forecast_df)
        except Exception as e:
            print(f"Error fetching data for {location_name}: {e}")
    
    # Combine all forecasts into a single DataFrame
    if all_forecasts:
        combined_df = pd.concat(all_forecasts, ignore_index=True)
        
        # Load existing CSV if it exists
        if os.path.exists(CSV_FILE):
            existing_df = pd.read_csv(CSV_FILE)
            existing_df['time'] = pd.to_datetime(existing_df['time'])
        else:
            existing_df = pd.DataFrame(columns=[
                'time', 'location', 'temperature', 'pressure', 
                'humidity', 'wind_speed', 'wind_direction', 
                'weather_icon', 'dew'
            ])
        
        # Merge new data with existing, keeping only new unique (time, location) records
        merged_df = pd.concat([existing_df, combined_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['time', 'location'], keep='last')
        
        # Sort by time and location for consistency
        merged_df = merged_df.sort_values(['time', 'location'])
        
        # Save to CSV
        merged_df.to_csv(CSV_FILE, index=False)
        print(f"Data saved to {CSV_FILE}. Total records: {len(merged_df)}")
    else:
        print("No data fetched from API.")