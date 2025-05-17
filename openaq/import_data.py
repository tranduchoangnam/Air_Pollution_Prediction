import pandas as pd
import sys
import os

# Add the parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timescaledb_util import TimescaleDBUtil

def import_aqi_data():
    """Import data from aqi_output.csv into TimescaleDB"""
    try:
        print("Starting import of AQI data...")
        
        # Read the CSV file
        csv_path = os.path.join(os.path.dirname(__file__), 'data/aqi_output.csv')
        print(f"Reading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} records in CSV file")
        
        # Loại bỏ bản ghi trùng lặp dựa trên location_id và datetimeLocal
        if 'location_id' in df.columns and 'datetimeLocal' in df.columns:
            df = df.drop_duplicates(subset=['location_id', 'datetimeLocal'])
            print(f"Số bản ghi sau khi loại bỏ trùng lặp: {len(df)}")
        
        # Convert datetime column to proper format
        df['datetimeLocal'] = pd.to_datetime(df['datetimeLocal'])
        
        # Initialize TimescaleDB connection
        print("Connecting to TimescaleDB...")
        db_url = 'postgres://tsdbadmin:lnmfese700b796cn@gejinnsvx3.aqgqm1fn3z.tsdb.cloud.timescale.com:35582/tsdb?sslmode=require'
        db_util = TimescaleDBUtil(db_url=db_url)
        
        # Xóa toàn bộ dữ liệu cũ trong bảng air_quality
        try:
            db_util.execute_query('DELETE FROM air_quality')
            print("Đã xóa toàn bộ dữ liệu cũ trong bảng air_quality.")
        except Exception as del_err:
            print(f"Không thể xóa dữ liệu cũ: {del_err}")
        
        # Check if table exists
        print("Checking if table exists...")
        table_exists = db_util.execute_query(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'air_quality')"
        ).iloc[0, 0]
        
        if not table_exists:
            print("Creating new table...")
            # Create new table
            success = db_util.create_table_from_dataframe(
                df=df,
                table_name='air_quality',
                time_column='datetimeLocal',
                schema='public',
                if_exists='replace'
            )
        else:
            print("Table already exists. Appending data...")
            # Insert data into existing table
            success = db_util.insert_dataframe(
                df=df,
                table_name='air_quality',
                schema='public',
                if_exists='append'
            )
        
        if success:
            print("Successfully imported all data to TimescaleDB")
            try:
                # Verify the data
                verify_query = """
                SELECT COUNT(*) as total_records,
                       MIN("datetimeLocal") as earliest_record,
                       MAX("datetimeLocal") as latest_record
                FROM air_quality
                """
                result = db_util.execute_query(verify_query)
                if result is not None:
                    print("\nVerification results:")
                    print(f"Total records: {result['total_records'].iloc[0]}")
                    print(f"Earliest record: {result['earliest_record'].iloc[0]}")
                    print(f"Latest record: {result['latest_record'].iloc[0]}")
                else:
                    print("Could not verify data - query returned no results")
            except Exception as verify_error:
                print(f"Error during verification: {str(verify_error)}")
        else:
            print("Failed to import data to TimescaleDB")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    import_aqi_data()
