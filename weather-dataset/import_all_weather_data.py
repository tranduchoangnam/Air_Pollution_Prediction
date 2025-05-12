import pandas as pd
import sys
import os

# Add the parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timescaledb_util import TimescaleDBUtil

def import_all_weather_data():
    """Import all data from hanoiweather_all.csv into TimescaleDB"""
    try:
        print("Starting import of all weather data...")
        
        # Read the CSV file
        csv_path = "weather-dataset/result/hanoiweather_all.csv"
        print(f"Reading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} records in CSV file")
        
        # Preprocess the data
        print("Preprocessing data...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['wind_speed'] = df['wind_speed'].str.replace(' km/h', '').astype(float)
        df['humidity'] = df['humidity'].str.replace('%', '').astype(float)
        df['temperature'] = df['temperature'].str.replace('°C', '').str.replace('°', '').astype(float)
        print("Data preprocessing completed")
        
        # Initialize TimescaleDB connection
        print("Connecting to TimescaleDB...")
        db_url = 'postgres://tsdbadmin:lnmfese700b796cn@gejinnsvx3.aqgqm1fn3z.tsdb.cloud.timescale.com:35582/tsdb?sslmode=require'
        db_util = TimescaleDBUtil(db_url=db_url)
        
        # Check if table exists
        print("Checking if table exists...")
        table_exists = db_util.execute_query(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'hanoi_weather_data')"
        ).iloc[0, 0]
        
        if not table_exists:
            print("Creating new table...")
            # Create new table
            success = db_util.create_table_from_dataframe(
                df=df,
                table_name='hanoi_weather_data',
                time_column='timestamp',
                schema='public',
                if_exists='replace'
            )
        else:
            print("Table already exists. Appending data...")
            # Insert data into existing table
            success = db_util.insert_dataframe(
                df=df,
                table_name='hanoi_weather_data',
                schema='public',
                if_exists='append'
            )
        
        if success:
            print("Successfully imported all data to TimescaleDB")
            # Verify the data
            verify_query = """
            SELECT COUNT(*) as total_records,
                   MIN(timestamp) as earliest_record,
                   MAX(timestamp) as latest_record
            FROM hanoi_weather_data
            """
            result = db_util.execute_query(verify_query)
            print("\nVerification results:")
            print(f"Total records: {result['total_records'].iloc[0]}")
            print(f"Earliest record: {result['earliest_record'].iloc[0]}")
            print(f"Latest record: {result['latest_record'].iloc[0]}")
        else:
            print("Failed to import data to TimescaleDB")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    import_all_weather_data() 