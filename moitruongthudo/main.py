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

# Function to transform crawled data to the format needed for TimescaleDB
def transform_to_df(crawl_results, output_data):
    # Pre-allocate a list with estimated size
    records = []
    
    # Flatten the nested structure in a more efficient way
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
                
                record = {
                    'time': time,
                    'data_type': dtype,
                    'station': station_name,
                    pollutant: value
                }
                records.append(record)
    
    # Only use DataFrame creation once at the end
    if records:
        df = pd.DataFrame(records)
        
        # Process all datetime conversions at once
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Process all numeric columns at once
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
        for col in pollutant_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Consolidate duplicate records (same time, station, data_type)
        df = df.groupby(['time', 'station', 'data_type'], as_index=False).first()
        
        return df
    return pd.DataFrame()

# Batch insert function for TimescaleDB
def batch_insert_to_timescale(df, table_name="air_quality_measurements", batch_size=1000, force_insert=False):
    if df.empty:
        return 0
        
    ts_db = TimescaleDBUtil()
    
    if not ts_db.connect():
        log("[!] Không thể kết nối tới TimescaleDB")
        return 0
    
    try:
        # Create table if it doesn't exist
        sample_df = df.head(1)
        ts_db.create_table_from_dataframe(
            df=sample_df,
            table_name=table_name,
            time_column="time",
            if_exists="append"
        )
        
        # If force_insert is True, skip the existence check
        if not force_insert:
            # Get existing records first to avoid checking one by one
            log("[ℹ] Kiểm tra các bản ghi đã tồn tại...")
            
            # Get all time, station, data_type combinations from DB
            existing_query = f"""
                SELECT time, station, data_type 
                FROM {table_name}
                WHERE time >= %s AND time <= %s
            """
            
            min_time = df['time'].min()
            max_time = df['time'].max()
            
            existing_df = ts_db.execute_query(existing_query, [min_time, max_time])
            
            if existing_df is not None and not existing_df.empty:
                # Create a set of tuples for fast lookup
                existing_records = set(
                    (row['time'], row['station'], row['data_type']) 
                    for _, row in existing_df.iterrows()
                )
                
                # Filter out existing records
                new_records = []
                for _, row in df.iterrows():
                    record_key = (row['time'], row['station'], row['data_type'])
                    if record_key not in existing_records:
                        new_records.append(row)
                        
                # Create new DataFrame with only new records
                if new_records:
                    df = pd.DataFrame(new_records)
                else:
                    df = pd.DataFrame(columns=df.columns)
        
        inserted_count = 0
        total_records = len(df)
        
        if not df.empty:
            # Insert in batches
            for i in range(0, total_records, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                success = ts_db.insert_dataframe(
                    df=batch_df,
                    table_name=table_name,
                    if_exists="append"
                )
                if success:
                    inserted_count += len(batch_df)
                    log(f"[✓] Đã thêm {inserted_count}/{total_records} bản ghi vào TimescaleDB")
        
        return inserted_count
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
    
    # If force_reinsert is true, we'll load all data from the JSON file
    if force_reinsert:
        log("[ℹ] Chế độ force reinsert được kích hoạt - sẽ thêm lại tất cả dữ liệu vào database")
        # Transform all data from output.json directly
        all_records = []
        
        for station_id, station_data in output_data.items():
            station_name = station_mapping.get(station_id, station_id)
            
            for data_type in ['stat', 'aqi']:
                pollutant_data = station_data.get(data_type, {})
                
                for pollutant, measurements in pollutant_data.items():
                    for entry in measurements:
                        time = entry.get('time')
                        value = entry.get('value') or entry.get(pollutant)
                        
                        record = {
                            'time': time,
                            'data_type': data_type,
                            'station': station_name,
                            pollutant: value
                        }
                        all_records.append(record)
        
        if all_records:
            df = pd.DataFrame(all_records)
            
            # Process all datetime conversions at once
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            
            # Process all numeric columns at once
            pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
            for col in pollutant_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            # Consolidate duplicate records (same time, station, data_type)
            df = df.groupby(['time', 'station', 'data_type'], as_index=False).first()
            
            # Insert all data with force_insert=True
            log(f"[ℹ] Chuẩn bị thêm {len(df)} bản ghi vào TimescaleDB...")
            inserted_count = batch_insert_to_timescale(df, force_insert=True)
            log(f"[✔] Đã thêm {inserted_count} bản ghi vào TimescaleDB")
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
    
    # Lưu kết quả vào file JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    log("[✔] Đã lưu dữ liệu phân cấp vào output.json")
    
    # Transform crawled data to DataFrame for TimescaleDB
    log("[ℹ] Chuyển đổi dữ liệu cho TimescaleDB...")
    df = transform_to_df(crawl_results, output_data)
    
    if not df.empty:
        # Insert data to TimescaleDB
        log(f"[ℹ] Chuẩn bị thêm {len(df)} bản ghi vào TimescaleDB...")
        inserted_count = batch_insert_to_timescale(df)
        log(f"[✔] Đã thêm {inserted_count} bản ghi mới vào TimescaleDB")
    else:
        log("[ℹ] Không có bản ghi mới để thêm vào TimescaleDB")
    
    log("===== Kết thúc crawl =====\n")

if __name__ == "__main__":
    main()