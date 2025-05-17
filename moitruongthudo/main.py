import requests
import json
import os
import sys
import pandas as pd
import concurrent.futures
from datetime import datetime
from collections import defaultdict

# Add the parent directory to the path for importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timescaledb_util import TimescaleDBUtil
from moitruongthudo.air_quality_prediction import AirQualityPrediction

# Các ID và địa điểm tương ứng
locations = {
    14: "số 46 lưu quang vũ",
    15: "minh khai bắc từ liêm",
    31: "đào duy từ",
    38: "an khánh",
    39: "vân hà"
}

# Station mapping cho hiển thị chuẩn hơn
station_mapping = {
    '14': '46 Lưu Quang Vũ',
    '15': 'Minh Khai - Bắc Từ Liêm',
    '31': 'Đào Duy Từ',
    '38': 'An Khánh',
    '39': 'Vân Hà'
}

# URL theo loại dữ liệu
data_types = {
    "stat": "https://moitruongthudo.vn/public/dailystat/{}",
    "aqi": "https://moitruongthudo.vn/public/dailyaqi/{}"
}

# File lưu trữ dữ liệu
base_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(base_dir, "output.json")
log_file = os.path.join(base_dir, "crawl_log.txt")

def log(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")

# Hàm kiểm tra dữ liệu đã tồn tại (dựa trên 'time')
def data_exists(data_list, new_item):
    return any(d['time'] == new_item['time'] for d in data_list)

# Hàm crawl dữ liệu từ một địa điểm và loại dữ liệu
def crawl_location_data(location_id, dtype, existing_data):
    loc_id_str = str(location_id)
    location_name = locations[location_id]
    url = data_types[dtype].format(location_id)
    
    new_records = {}
    new_records_count = 0
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Dữ liệu có thể gồm nhiều key như "PM2.5", "PM10",...
        for key in data:
            if key not in existing_data.get(loc_id_str, {}).get(dtype, {}):
                if key not in new_records:
                    new_records[key] = []
                new_records[key] = data[key]
                new_records_count += len(data[key])
            else:
                if key not in new_records:
                    new_records[key] = []
                for item in data[key]:
                    if not data_exists(existing_data[loc_id_str][dtype][key], item):
                        new_records[key].append(item)
                        new_records_count += 1
        
        return {
            "location_id": loc_id_str,
            "location_name": location_name,
            "dtype": dtype,
            "new_records": new_records,
            "count": new_records_count,
            "success": True
        }
    except Exception as e:
        return {
            "location_id": loc_id_str,
            "location_name": location_name,
            "dtype": dtype,
            "error": str(e),
            "success": False
        }

# Process crawled data to insert into mttd_aqi table
def process_for_mttd_aqi(crawl_results):
    """Transform crawled data to format needed for mttd_aqi table with AQI calculations"""
    # Initialize AirQualityPrediction instance for AQI calculation
    aqi_calculator = AirQualityPrediction()
    
    # Create a map to store data by station and timestamp
    # This helps us merge 'stat' and 'aqi' data
    station_time_data = defaultdict(dict)
    
    # First process all the 'stat' data
    for result in crawl_results:
        if not result["success"]:
            continue
            
        station_id = result["location_id"]
        dtype = result["dtype"]
        station_name = station_mapping.get(station_id, station_id)
        
        for pollutant, measurements in result["new_records"].items():
            for entry in measurements:
                time = entry.get('time')
                value = entry.get('value') or entry.get(pollutant)
                
                # Skip if value is None
                if value is None:
                    continue
                
                # Try to convert value to float immediately
                try:
                    if isinstance(value, str):
                        value = float(value)
                except (ValueError, TypeError):
                    # If conversion fails, keep as is and handle later
                    pass
                
                # Create unique key for station and time
                key = f"{station_name}_{time}"
                
                # Initialize record if needed
                if key not in station_time_data:
                    station_time_data[key] = {
                        'time': time,
                        'station': station_name,
                        'data_type': {}  # Store both 'stat' and 'aqi' data
                    }
                
                # Store the data by type
                if dtype not in station_time_data[key]['data_type']:
                    station_time_data[key]['data_type'][dtype] = {}
                
                # Map the pollutant name to column name in mttd_aqi
                column_name = pollutant
                if pollutant == "PM2.5":
                    column_name = "PM2_5"
                
                # Store the value
                station_time_data[key]['data_type'][dtype][column_name] = value
    
    # Create records for mttd_aqi table
    records = []
    
    for key, data in station_time_data.items():
        # Get values preferring 'stat' but falling back to 'aqi'
        pollutants = {
            'PM2_5': None, 'PM10': None, 'O3': None, 
            'SO2': None, 'NO2': None, 'CO': None
        }
        
        # Only get concentration values from 'stat' data
        if 'stat' in data['data_type']:
            stat_data = data['data_type']['stat']
            for pollutant in pollutants:
                if pollutant in stat_data:
                    pollutants[pollutant] = stat_data[pollutant]
        
        # Calculate AQI values for each pollutant from 'stat' data only
        aqi_values = {}
        for pollutant, value in pollutants.items():
            if value is not None:
                try:
                    # Convert value to float before calculation
                    numeric_value = float(value)
                    
                    # Map column name to pollutant name expected by calculate_aqi
                    aqi_pollutant = pollutant.lower()
                    if pollutant == 'PM2_5':
                        aqi_pollutant = 'pm25'
                    
                    aqi_value = aqi_calculator.calculate_aqi(numeric_value, aqi_pollutant)
                    aqi_column = f"{pollutant}_AQI"
                    aqi_values[aqi_column] = aqi_value
                except (ValueError, TypeError) as e:
                    # Log the error and continue with other pollutants
                    log(f"[!] Error calculating AQI for {pollutant}: {str(e)}")
        
        # Fill missing AQI values directly from 'aqi' data without recalculating
        if 'aqi' in data['data_type']:
            aqi_data = data['data_type']['aqi']
            for pollutant in pollutants:
                aqi_column = f"{pollutant}_AQI"
                # If we don't have an AQI value for this pollutant yet, use the value from 'aqi' data directly
                if (aqi_column not in aqi_values or aqi_values[aqi_column] is None) and pollutant in aqi_data:
                    try:
                        # Use AQI value directly from 'aqi' data
                        aqi_value = float(aqi_data[pollutant])
                        aqi_values[aqi_column] = aqi_value
                    except (ValueError, TypeError) as e:
                        log(f"[!] Error using AQI value for {pollutant} from 'aqi' data: {str(e)}")
        
        # Calculate overall AQI
        overall_aqi = None
        if aqi_values:
            overall_aqi = max(v for v in aqi_values.values() if v is not None)
        
        # Calculate AQI category
        aqi_category = None
        if overall_aqi is not None:
            aqi_category = aqi_calculator.get_aqi_category(overall_aqi)
        
        # Create the record
        record = {
            'time': data['time'],
            'station': data['station'],
            **pollutants,
            **aqi_values,
            'AQI': overall_aqi,
            'AQI_category': aqi_category
        }
        
        records.append(record)
    
    # Convert to DataFrame
    if records:
        df = pd.DataFrame(records)
        
        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['PM2_5', 'PM10', 'O3', 'SO2', 'NO2', 'CO', 
                        'PM2_5_AQI', 'PM10_AQI', 'O3_AQI', 'SO2_AQI', 'NO2_AQI', 'CO_AQI', 'AQI']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    return pd.DataFrame()

# Insert data into TimescaleDB mttd_aqi table
def insert_to_mttd_aqi(df, batch_size=1000):
    """Insert DataFrame into mttd_aqi table, avoiding duplicates and adding any missing columns automatically"""
    if df.empty:
        log("[ℹ] No data to insert into mttd_aqi table")
        return 0
    
    ts_db = TimescaleDBUtil()
    
    if not ts_db.connect():
        log("[!] Cannot connect to TimescaleDB")
        return 0
    
    try:
        # Log the total number of records to process
        total_records = len(df)
        log(f"[ℹ] Processing {total_records} records for insertion into mttd_aqi table")
        
        # Check for existing records to avoid duplicates
        start_time = datetime.now()
        log(f"[ℹ] Checking for existing records (start time: {start_time.strftime('%H:%M:%S')})...")
        
        # Filter out records that already exist in the database
        filtered_df = ts_db.filter_existing_records(
            df=df,
            table_name="mttd_aqi",
            time_column="time",
            additional_columns=["station"]
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if filtered_df is None:
            log(f"[!] Error checking for existing records (duration: {duration:.2f} seconds)")
            return 0
        
        if filtered_df.empty:
            log(f"[ℹ] All records already exist in the database (duration: {duration:.2f} seconds)")
            return 0
        
        # Insert new records with automatic column adaptation
        records_count = len(filtered_df)
        log(f"[ℹ] Found {records_count} new records to insert (duration: {duration:.2f} seconds)")
        log(f"[ℹ] Inserting {records_count} new records into mttd_aqi table with automatic schema adaptation...")
        
        # Use the new flexible_insert_dataframe method that automatically handles missing columns
        start_time = datetime.now()
        success = ts_db.flexible_insert_dataframe(
            df=filtered_df,
            table_name="mttd_aqi",
            if_exists="append",
            batch_size=batch_size
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if success:
            log(f"[✓] Successfully added {records_count} records to mttd_aqi table (duration: {duration:.2f} seconds)")
            return records_count
        else:
            log(f"[!] Failed to insert records into mttd_aqi table (duration: {duration:.2f} seconds)")
            return 0
            
    except Exception as e:
        log(f"[!] Error inserting data into mttd_aqi table: {str(e)}")
        return 0
    finally:
        ts_db.disconnect()

# Main execution
def main():
    # Check for force reinsert flag
    force_reinsert = "--force" in sys.argv
    
    # Load dữ liệu cũ nếu có
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)
    else:
        output_data = {}
    
    log("===== Bắt đầu crawl =====")
    
    # If force_reinsert is true, we'll process all data from the JSON file
    if force_reinsert:
        log("[ℹ] Chế độ force reinsert được kích hoạt - sẽ xử lý lại tất cả dữ liệu từ output.json")
        
        # Create artificial crawl results from output.json
        force_crawl_results = []
        
        for station_id, station_data in output_data.items():
            station_name = station_mapping.get(station_id, station_id)
            
            for dtype in ['stat', 'aqi']:
                if dtype in station_data:
                    # Create a crawl result object for each data type
                    result = {
                        "location_id": station_id,
                        "location_name": station_name,
                        "dtype": dtype,
                        "new_records": station_data[dtype],
                        "count": sum(len(items) for items in station_data[dtype].values()),
                        "success": True
                    }
                    force_crawl_results.append(result)
        
        # Process data from output.json
        log("[ℹ] Đang xử lý dữ liệu cho bảng mttd_aqi từ output.json...")
        df = process_for_mttd_aqi(force_crawl_results)
        
        if not df.empty:
            # Insert data to mttd_aqi table
            log(f"[ℹ] Chuẩn bị thêm {len(df)} bản ghi vào bảng mttd_aqi...")
            inserted_count = insert_to_mttd_aqi(df)
            log(f"[✔] Đã thêm {inserted_count} bản ghi mới vào bảng mttd_aqi")
        else:
            log("[ℹ] Không có bản ghi để thêm vào bảng mttd_aqi từ output.json")
        
        log("===== Kết thúc crawl =====\n")
        return
    
    # Prepare crawl tasks
    crawl_tasks = []
    for location_id in locations:
        loc_id_str = str(location_id)
        
        # Initialize data structure if needed
        if loc_id_str not in output_data:
            output_data[loc_id_str] = {}
            
        for dtype in data_types:
            if dtype not in output_data.get(loc_id_str, {}):
                output_data[loc_id_str][dtype] = {}
                
            crawl_tasks.append((location_id, dtype, output_data))
    
    # Execute crawl tasks in parallel
    crawl_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {
            executor.submit(crawl_location_data, task[0], task[1], task[2]): task 
            for task in crawl_tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                crawl_results.append(result)
                
                # Log result
                if result["success"]:
                    log(f"[✓] {result['dtype'].upper()} - ID {result['location_id']} ({result['location_name']}): +{result['count']} bản ghi mới")
                else:
                    log(f"[!] Lỗi khi crawl {result['dtype']} từ ID {result['location_id']}: {result['error']}")
                    
                # Update output_data
                if result["success"] and result["count"] > 0:
                    loc_id = result["location_id"]
                    dtype = result["dtype"]
                    
                    if loc_id not in output_data:
                        output_data[loc_id] = {}
                    if dtype not in output_data[loc_id]:
                        output_data[loc_id][dtype] = {}
                        
                    for key, items in result["new_records"].items():
                        if key not in output_data[loc_id][dtype]:
                            output_data[loc_id][dtype][key] = []
                        
                        # Only add items that don't exist
                        for item in items:
                            if not data_exists(output_data[loc_id][dtype].get(key, []), item):
                                output_data[loc_id][dtype].setdefault(key, []).append(item)
                
            except Exception as e:
                location_id, dtype, _ = task
                log(f"[!] Exception in crawl task for ID {location_id}, type {dtype}: {str(e)}")
    
    # Lưu kết quả vào file JSON (vẫn giữ để backup)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    log("[✔] Đã lưu dữ liệu phân cấp vào output.json")
    
    # Process data for mttd_aqi table
    log("[ℹ] Đang xử lý dữ liệu cho bảng mttd_aqi...")
    df = process_for_mttd_aqi(crawl_results)
    
    if not df.empty:
        # Insert data to mttd_aqi table
        log(f"[ℹ] Chuẩn bị thêm {len(df)} bản ghi vào bảng mttd_aqi...")
        inserted_count = insert_to_mttd_aqi(df)
        log(f"[✔] Đã thêm {inserted_count} bản ghi mới vào bảng mttd_aqi")
    else:
        log("[ℹ] Không có bản ghi mới để thêm vào bảng mttd_aqi")
    
    log("===== Kết thúc crawl =====\n")

if __name__ == "__main__":
    main()