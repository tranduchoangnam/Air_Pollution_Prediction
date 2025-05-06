from playwright.sync_api import sync_playwright
import json
from datetime import datetime
import csv
import os
import pathlib
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional
import re

# Đọc dữ liệu từ file location.csv
def load_locations_from_csv(file_path: str) -> List[Dict[str, str]]:
    locations = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            location = {
                "url": row['url'],
                "name": row['name']
            }
            locations.append(location)
    return locations

# Lấy dữ liệu từ file location.csv
file_path = 'weather-dataset/location.csv'  
LOCATIONS = load_locations_from_csv(file_path)

def get_vietnam_time():
    """Get current time in Vietnam timezone (GMT+7) without microseconds"""
    return datetime.now(ZoneInfo("Asia/Bangkok")).replace(microsecond=0)

def validate_weather_icon(icon: str) -> Optional[str]:
    """Validate weather icon URL and convert it to a readable description"""
    if icon and isinstance(icon, str):
        icon_filename = icon.split('/')[-1]  # Extract the filename if it's a path

        # Mapping from icon filenames to readable descriptions
        icon_mapping = {
            "ic-w-01-clear-sky-full.svg": "clear-day",
            "ic-w-01-clear-day-full.svg": "clear-day",
            "ic-w-02-new-clouds-full.svg": "partly-cloudy-day",
            "ic-w-02-wind-full.svg": "wind",
            "ic-w-03-cloudy-full.svg": "cloudy",
            "ic-w-04-scattered-clouds-full.svg": "partly-cloudy-day",
            "ic-w-06-rain-full.svg": "rain",
            "ic-w-07-thunderstorms-full.svg": "thunderstorm",
            "ic-w-09-mist-full.svg": "mist",
            "ic-w-10-night-clear-sky-full.svg": "clear-night",
            "ic-w-11-night-few-clouds-full.svg": "partly-cloudy-night",
            "ic-w-12-night-rain-full.svg": "rain",

            # fallback for existing readable descriptions already in data
            "partly-cloudy-day": "partly-cloudy-day",
            "rain": "rain",
            "cloudy": "cloudy",
            "mist": "mist",
            "wind": "wind",
            "clear-day": "clear-day"
        }

        return icon_mapping.get(icon_filename, icon_filename)
    return None

def validate_wind_speed(speed: str) -> Optional[str]:
    """Validate wind speed"""
    try:
        # Check if matches pattern like "10.2 km/h" or "8.5 mph"
        if re.match(r'^\d+(\.\d+)?\s*(km/h|mph)$', speed.strip()):
            # Convert mph to km/h if needed
            speed = speed.strip()
            if 'mph' in speed:
                # Extract numeric value
                value = float(re.match(r'^\d+(\.\d+)?', speed).group())
                # Convert to km/h (1 mile = 1.60934 kilometers)
                km_value = value * 1.60934
                return f"{km_value:.1f} km/h"
            return speed
    except (ValueError, TypeError, AttributeError):
        pass
    return None

def validate_humidity(humidity: str) -> Optional[str]:
    """Validate humidity"""
    try:
        # Check if matches pattern like "39%"
        if re.match(r'^\d{1,3}%$', humidity.strip()):
            return humidity.strip()
    except (ValueError, TypeError, AttributeError):
        pass
    return None

def validate_temperature(temp: str) -> Optional[str]:
    """Validate and convert temperature to Celsius"""
    try:
        temp = temp.strip()
        print(f"Raw temperature: {temp}")  # Debug log
        
        # Extract numeric value and unit
        match = re.search(r'(\d+)\s*°?\s*([CF])?', temp, re.IGNORECASE)
        if not match:
            return None
            
        value = int(match.group(1))
        unit = match.group(2).upper() if match.group(2) else None
        
        # Convert to Celsius if needed
        if unit == 'F' or (unit is None and value > 50):  # Assume F if >50 and no unit
            celsius = round((value - 32) * 5/9)
            return f"{celsius}°C"
        
        return f"{value}°C"
        
    except Exception as e:
        print(f"Temperature validation error: {str(e)}")
        return None

def crawl_location_data(page, location: Dict) -> Optional[Dict]:
    """Crawl data for a specific location"""
    print(f"\nAccessing {location['name']} ({location['url']})...")
    
    try:
        # Navigate to location page
        page.goto(location['url'])
        
        # Wait for content to load
        page.wait_for_selector(".aqi-value__estimated")
        
        # Extract and validate data
        weather_icon_raw = page.query_selector(".air-quality-forecast-container-weather__icon").get_attribute("src")
        wind_speed_raw = page.query_selector(".air-quality-forecast-container-wind__label").text_content()
        humidity_raw = page.query_selector(".air-quality-forecast-container-humidity__label").text_content()
        
        # Assuming the temperature is in a specific selector like '.air-quality-forecast-container-weather__label'
        temperature_raw = page.query_selector(".air-quality-forecast-container-weather__label").text_content()

        # Validate all fields
        weather_icon = validate_weather_icon(weather_icon_raw)
        wind_speed = validate_wind_speed(wind_speed_raw)
        humidity = validate_humidity(humidity_raw)
        temperature = temperature = validate_temperature(temperature_raw)
        
        # If any validation fails, return None
        if not all([weather_icon, wind_speed, humidity, temperature]):
            print(f"Invalid data found for {location['name']}:")
            if not weather_icon: print(f"  - Invalid weather icon: {weather_icon_raw}")
            if not wind_speed: print(f"  - Invalid wind speed: {wind_speed_raw}")
            if not humidity: print(f"  - Invalid humidity: {humidity_raw}")
            if not temperature: print(f"  - Invalid temperature: {temperature_raw}")
            return None
        
        # Create data dictionary with Vietnam time
        current_time = get_vietnam_time()
        data = {
            "timestamp": current_time.isoformat(),
            "location": location['name'],
            "weather_icon": weather_icon,
            "wind_speed": wind_speed,
            "humidity": humidity,
            "temperature": temperature  
        }
        
        return data
        
    except Exception as e:
        print(f"Error extracting data for {location['name']}: {str(e)}")
        return None

def save_to_csv(data: Dict, location_name: str):
    """Save data to CSV file for a specific location"""
    now = get_vietnam_time()
    result_dir = pathlib.Path(f"weather-dataset/result")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename based on current month
    filename = f"hanoiweather_all.csv"
    filepath = result_dir / filename
    
    # Define CSV headers
    headers = ["timestamp", "weather_icon", "wind_speed", "humidity", "temperature", "location"]
    
    # Check if file exists to determine if we need to write headers
    file_exists = filepath.exists()
    
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        
        # Write headers if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data
        writer.writerow(data)
    
    return filepath

def crawl_all_locations():
    """Crawl data for all locations"""
    results = []
    for location in LOCATIONS:
        print(f"\n{'='*50}")
        print(f"Processing {location['name']}...")
        try:
            with sync_playwright() as p:
                try:
                    # Launch new browser for each location
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    
                    # Set viewport size for better rendering
                    page.set_viewport_size({"width": 1280, "height": 720})
                    
                    # Add small delay for stability
                    page.set_default_timeout(15000)  # 15 seconds timeout
                    
                    data = crawl_location_data(page, location)
                    if data:  # Only process valid data
                        data["location"] = location["name"]
                        results.append(data)
                        # Save to CSV
                        csv_file = save_to_csv(data, location['name'])
                        print(f"Data saved to: {csv_file}")
                    else:
                        print(f"Skipping invalid data for {location['name']}")
                
                except Exception as e:
                    print(f"Browser error for {location['name']}: {str(e)}")
                    continue
                
                finally:
                    if 'browser' in locals():
                        browser.close()
        
        except Exception as e:
            print(f"Playwright error for {location['name']}: {str(e)}")
            continue
            
    return results

if __name__ == "__main__":
    try:
        print("Starting IQAir data crawler...")
        print(f"Current time in Vietnam: {get_vietnam_time().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        results = crawl_all_locations()
        
        print("\nCrawled data:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e
