import os
import psycopg2
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
try:
    from sqlalchemy import create_engine
    # Check if psycopg2 is properly installed
    import psycopg2
    # Try to explicitly import the postgresql dialect
    import sqlalchemy.dialects.postgresql
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Try installing required packages with: pip install sqlalchemy psycopg2-binary")

class TimescaleDBUtil:
    """A reusable utility class for TimescaleDB operations"""
    
    def __init__(self, db_url=None, env_path=None):
        """Initialize with either a direct connection URL or path to an .env file"""
        self.db_params, self.db_url = self._get_connection_params(db_url, env_path)
        self.conn = None
        
    def _get_connection_params(self, db_url=None, env_path=None):
        """Get connection parameters from DB_URL or .env file"""
        # If DB URL is provided directly, use it
        if db_url:
            result = urlparse(db_url)
            db_params = {
                'dbname': result.path[1:],
                'user': result.username,
                'password': result.password,
                'host': result.hostname,
                'port': result.port
            }
            return db_params, db_url
        
        # Try to load from .env file
        if not env_path:
            # Try to find .env in the parent directory
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('DB_URL='):
                        db_url = line.strip().split('DB_URL=')[1]
                        # Parse the connection URL
                        result = urlparse(db_url)
                        db_params = {
                            'dbname': result.path[1:],
                            'user': result.username,
                            'password': result.password,
                            'host': result.hostname,
                            'port': result.port
                        }
                        return db_params, db_url
        
        # Fallback to default values if specified in environment variables
        if 'DB_URL' in os.environ:
            db_url = os.environ['DB_URL']
            result = urlparse(db_url)
            db_params = {
                'dbname': result.path[1:],
                'user': result.username,
                'password': result.password,
                'host': result.hostname,
                'port': result.port
            }
            return db_params, db_url
        
        # Hardcoded fallback (not recommended for production)
        db_url = 'postgres://tsdbadmin:lnmfese700b796cn@gejinnsvx3.aqgqm1fn3z.tsdb.cloud.timescale.com:35582/tsdb?sslmode=require'
        result = urlparse(db_url)
        db_params = {
            'dbname': result.path[1:],
            'user': result.username,
            'password': result.password,
            'host': result.hostname,
            'port': result.port
        }
        return db_params, db_url
    
    def connect(self):
        """Establish a connection to TimescaleDB"""
        try:
            self.conn = psycopg2.connect(**self.db_params, sslmode='require')
            return True
        except Exception as e:
            print(f"Error connecting to TimescaleDB: {e}")
            return False
    
    def disconnect(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_hypertable(self, table_name, time_column, schema="public"):
        """Create a hypertable in TimescaleDB"""
        if not self.conn:
            if not self.connect():
                return False
        
        try:
            cursor = self.conn.cursor()
            
            # Enable TimescaleDB extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # Create the hypertable with migrate_data option
            cursor.execute(f"""
                SELECT create_hypertable('{schema}.{table_name}', '{time_column}', 
                                         if_not_exists => TRUE,
                                         migrate_data => TRUE);
            """)
            
            self.conn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error creating hypertable: {e}")
            self.conn.rollback()
            return False
    
    def create_table_from_dataframe(self, df, table_name, time_column, schema="public", if_exists="replace"):
        """Create a table from a pandas DataFrame and convert it to a hypertable"""
        if not self.conn:
            if not self.connect():
                return False
        
        try:
            # First create the table using pandas
            from sqlalchemy import create_engine
            
            # Ensure SQLAlchemy connection string is properly formatted
            if 'postgresql://' not in self.db_url and 'postgres://' in self.db_url:
                # Convert postgres:// to postgresql:// for SQLAlchemy
                sqlalchemy_url = self.db_url.replace('postgres://', 'postgresql://')
            else:
                sqlalchemy_url = self.db_url
                
            engine = create_engine(sqlalchemy_url)
            
            # Ensure time column is in datetime format
            if time_column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                df[time_column] = pd.to_datetime(df[time_column])
            
            # Create the table
            df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=False)
            
            # Convert to hypertable
            return self.create_hypertable(table_name, time_column, schema)
        except Exception as e:
            print(f"Error creating table from DataFrame: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """Execute a SQL query and return the results"""
        if not self.conn:
            if not self.connect():
                return None
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            # Try to fetch results if it's a SELECT query
            try:
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                cursor.close()
                return pd.DataFrame(results, columns=column_names)
            except:
                # For non-SELECT queries
                self.conn.commit()
                cursor.close()
                return True
        except Exception as e:
            print(f"Error executing query: {e}")
            self.conn.rollback()
            return None
    
    def execute_query_from_file(self, sql_file_path, params=None):
        """Execute SQL from a file"""
        try:
            with open(sql_file_path, 'r') as f:
                sql = f.read()
            return self.execute_query(sql, params)
        except Exception as e:
            print(f"Error reading SQL file: {e}")
            return None
    
    def insert_dataframe(self, df, table_name, schema="public", if_exists="append"):
        """Insert a DataFrame into an existing table"""
        try:
            # Use explicit postgresql:// URL format
            if 'postgresql://' not in self.db_url and 'postgres://' in self.db_url:
                # Convert postgres:// to postgresql:// for SQLAlchemy
                sqlalchemy_url = self.db_url.replace('postgres://', 'postgresql://')
            else:
                sqlalchemy_url = self.db_url
                
            # Create the SQLAlchemy engine with explicit dialect
            try:
                engine = create_engine(sqlalchemy_url)
                df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=False)
                return True
            except Exception as e:
                print(f"Error with SQLAlchemy engine: {e}")
                
                # Try alternate method using psycopg2 directly for insert
                if not self.conn:
                    if not self.connect():
                        return False
                
                cursor = self.conn.cursor()
                
                # Get column names
                columns = df.columns.tolist()
                col_str = ", ".join([f'"{col}"' for col in columns])
                
                # Prepare values
                placeholders = ", ".join(["%s"] * len(columns))
                
                # Build query
                query = f'INSERT INTO {schema}.{table_name} ({col_str}) VALUES ({placeholders})'
                
                # Execute for each row
                for _, row in df.iterrows():
                    values = [row[col] for col in columns]
                    cursor.execute(query, values)
                
                self.conn.commit()
                cursor.close()
                return True
                
        except Exception as e:
            print(f"Error inserting DataFrame: {e}")
            return False
    
    # Add a new flexible method to insert DataFrame with automatic schema adaptation
    def flexible_insert_dataframe(self, df, table_name, schema="public", if_exists="append", batch_size=1000, on_conflict=None):
        """
        Insert a DataFrame into a table, automatically adding any missing columns.
        
        Args:
            df: DataFrame to insert
            table_name: Name of target table
            schema: Database schema
            if_exists: How to behave if the table exists ("append" or "replace")
            batch_size: Number of records to insert in a single batch
            on_conflict: Optional conflict handling clause (e.g., "ON CONFLICT (time, location) DO NOTHING")
            
        Returns:
            Boolean indicating success or failure
        """
        if df.empty:
            print("DataFrame is empty, nothing to insert")
            return False
        
        if not self.conn:
            if not self.connect():
                return False
                
        try:
            # Check if table exists, create it if necessary
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT to_regclass('{schema}.{table_name}')")
            table_exists = cursor.fetchone()[0] is not None
            
            # If table doesn't exist, create it
            if not table_exists:
                print(f"Table {schema}.{table_name} doesn't exist, creating it")
                # Use standard insert_dataframe which will create the table
                return self.insert_dataframe(df.head(1), table_name, schema, if_exists)
            
            # Get existing columns
            cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='{schema}' AND table_name='{table_name}'")
            existing_columns = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Check for missing columns
            missing_columns = []
            for col in df.columns:
                if col not in existing_columns:
                    missing_columns.append(col)
            
            # Add missing columns if necessary
            if missing_columns:
                print(f"Adding {len(missing_columns)} missing columns to {schema}.{table_name}: {missing_columns}")
                for col in missing_columns:
                    # Determine appropriate PostgreSQL type based on DataFrame dtype
                    col_type = None
                    dtype = df[col].dtype
                    
                    # Handle different pandas dtypes
                    if pd.api.types.is_integer_dtype(dtype):
                        col_type = "BIGINT"
                    elif pd.api.types.is_float_dtype(dtype):
                        col_type = "DOUBLE PRECISION"
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        col_type = "TIMESTAMP"
                    elif pd.api.types.is_bool_dtype(dtype):
                        col_type = "BOOLEAN"
                    else:
                        # Default to TEXT for strings and other types
                        col_type = "TEXT"
                        
                    # If column name ends with _AQI, use BIGINT
                    if col.endswith('_AQI'):
                        col_type = "BIGINT"
                    
                    # Add the column
                    alter_query = f'ALTER TABLE {schema}.{table_name} ADD COLUMN IF NOT EXISTS "{col}" {col_type};'
                    cursor.execute(alter_query)
                    print(f"Added column {col} with type {col_type}")
                
                self.conn.commit()
            
            # Now insert the data
            # Get column names
            columns = df.columns.tolist()
            col_str = ", ".join([f'"{col}"' for col in columns])
            
            # Prepare values
            placeholders = ", ".join(["%s"] * len(columns))
            
            # Build query
            query = f'INSERT INTO {schema}.{table_name} ({col_str}) VALUES ({placeholders})'
            
            # Add ON CONFLICT clause if provided
            if on_conflict:
                query += f" {on_conflict}"
            
            # Execute in batches using executemany for better performance
            rows_inserted = 0
            total_rows = len(df)
            
            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                # Create a list of tuples for executemany
                batch_values = []
                for _, row in batch_df.iterrows():
                    values = tuple(None if pd.isna(v) else v for v in row)
                    batch_values.append(values)
                
                # Use executemany for better performance with batches
                if batch_values:
                    cursor.executemany(query, batch_values)
                    self.conn.commit()
                    
                rows_inserted += len(batch_values)
                print(f"Inserted {rows_inserted}/{total_rows} rows ({(rows_inserted/total_rows*100):.1f}%)")
            
            cursor.close()
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error in flexible_insert_dataframe: {e}")
            return False
    
    # Add more utility methods as needed for specific use cases
    
    def get_latest_data(self, table_name, time_column="time", limit=100, schema="public"):
        """Get the latest data from a hypertable"""
        query = f"""
            SELECT * FROM {schema}.{table_name}
            ORDER BY {time_column} DESC
            LIMIT {limit};
        """
        return self.execute_query(query)
    
    def get_time_bucket_data(self, table_name, time_column, interval, 
                           aggregations, where_clause="", group_by="", 
                           order_by="time_bucket", schema="public"):
        """Get data aggregated by time buckets
        
        Args:
            table_name: The name of the table
            time_column: The name of the time column
            interval: Time bucket interval (e.g., '1 hour', '1 day')
            aggregations: List of aggregation expressions (e.g., ['AVG(temperature)', 'MAX(humidity)'])
            where_clause: Optional WHERE clause
            group_by: Optional additional GROUP BY columns
            order_by: Order by clause
            schema: Database schema
        """
        agg_columns = ", ".join(aggregations)
        group_by_clause = f", {group_by}" if group_by else ""
        where_clause = f"WHERE {where_clause}" if where_clause else ""
        
        query = f"""
            SELECT 
                time_bucket('{interval}', {time_column}) AS time_bucket
                {group_by_clause},
                {agg_columns}
            FROM {schema}.{table_name}
            {where_clause}
            GROUP BY time_bucket {group_by_clause}
            ORDER BY {order_by};
        """
        
        return self.execute_query(query)
    
    def check_record_exists(self, table_name, time_column, time_value, additional_conditions=None, schema="public"):
        """Check if a record exists in the table based on time and optional additional conditions
        
        Args:
            table_name: The name of the table to check
            time_column: The name of the time column
            time_value: The time value to check for (datetime object or string)
            additional_conditions: Dictionary of column_name: value pairs for additional conditions
            schema: Database schema
            
        Returns:
            Boolean indicating if the record exists
        """
        if not self.conn:
            if not self.connect():
                return False
        
        try:
            cursor = self.conn.cursor()
            
            # Convert time_value to proper format if it's a datetime
            if isinstance(time_value, datetime):
                time_str = time_value.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = time_value
                
            # Build the query
            query = f"SELECT 1 FROM {schema}.{table_name} WHERE {time_column} = %s"
            params = [time_str]
            
            # Add additional conditions if provided
            if additional_conditions:
                for col, val in additional_conditions.items():
                    query += f" AND {col} = %s"
                    params.append(val)
                    
            # Execute the query
            cursor.execute(query, params)
            result = cursor.fetchone()
            cursor.close()
            
            # Return True if record exists, False otherwise
            return result is not None
            
        except Exception as e:
            print(f"Error checking record existence: {e}")
            return False
            
    def filter_existing_records(self, df, table_name, time_column, additional_columns=None, schema="public"):
        """Filter out records that already exist in the database using an optimized batch approach
        
        Args:
            df: DataFrame containing records to check
            table_name: The name of the table to check against
            time_column: The name of the time column
            additional_columns: List of additional column names to use for uniqueness check
            schema: Database schema
            
        Returns:
            DataFrame with only new records
        """
        if df.empty:
            return df
            
        if not self.conn:
            if not self.connect():
                return df
        
        try:
            # Ensure time_column is properly formatted
            if pd.api.types.is_datetime64_any_dtype(df[time_column]):
                # Convert timestamps to strings for SQL comparison
                df['time_str'] = df[time_column].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # If it's already a string or other format
                df['time_str'] = df[time_column].astype(str)
            
            # Create a temporary table with records to check
            temp_table = f"temp_check_{table_name}_{int(datetime.now().timestamp())}"
            
            cursor = self.conn.cursor()
            
            # Define columns for the temporary table
            check_columns = ['time_str']
            if additional_columns:
                check_columns.extend(additional_columns)
            
            # Create SQL for temporary table columns
            col_definitions = ['"time_str" TEXT']
            if additional_columns:
                for col in additional_columns:
                    col_definitions.append(f'"{col}" TEXT')
            
            # Create temporary table to hold records we need to check
            create_temp_sql = f"""
                CREATE TEMPORARY TABLE {temp_table} (
                    {', '.join(col_definitions)}
                )
            """
            cursor.execute(create_temp_sql)
            
            # Insert records to check into temporary table
            values = []
            placeholders = []
            
            # Build batch insert SQL
            for idx, row in df.iterrows():
                row_values = [row['time_str']]
                if additional_columns:
                    for col in additional_columns:
                        if col in row and pd.notna(row[col]):
                            row_values.append(str(row[col]))
                        else:
                            row_values.append(None)
                
                values.extend(row_values)
                ph = '(' + ', '.join(['%s'] * len(row_values)) + ')'
                placeholders.append(ph)
            
            # Insert in one batch
            if placeholders:
                insert_sql = f"""
                    INSERT INTO {temp_table} 
                    VALUES {', '.join(placeholders)}
                """
                cursor.execute(insert_sql, values)
            
            # Build query to find existing records
            join_conditions = [f't.time_str = CAST(e.{time_column} AS TEXT)']
            if additional_columns:
                for col in additional_columns:
                    join_conditions.append(f't."{col}" = CAST(e."{col}" AS TEXT)')
            
            # Find records that already exist
            exists_query = f"""
                SELECT t.time_str 
                {', t.' + ', t.'.join(f'"{col}"' for col in additional_columns) if additional_columns else ''}
                FROM {temp_table} t
                JOIN {schema}.{table_name} e
                ON {' AND '.join(join_conditions)}
            """
            
            cursor.execute(exists_query)
            existing_records = cursor.fetchall()
            
            # Clean up temporary table
            cursor.execute(f"DROP TABLE {temp_table}")
            self.conn.commit()
            
            # Convert results to a set of tuples for fast lookup
            existing_set = set()
            for record in existing_records:
                existing_set.add(tuple(str(v) if v is not None else None for v in record))
            
            # Filter out existing records
            if existing_set:
                mask = []
                for idx, row in df.iterrows():
                    key = [row['time_str']]
                    if additional_columns:
                        for col in additional_columns:
                            if col in row and pd.notna(row[col]):
                                key.append(str(row[col]))
                            else:
                                key.append(None)
                    
                    # If record exists, exclude it (False), otherwise include it (True)
                    mask.append(tuple(key) not in existing_set)
                
                # Apply the mask to get only new records
                new_df = df[mask].copy()
            else:
                # No existing records found, so all are new
                new_df = df.copy()
            
            # Remove the temporary time_str column
            if 'time_str' in new_df.columns:
                new_df = new_df.drop(columns=['time_str'])
            
            return new_df
            
        except Exception as e:
            print(f"Error filtering existing records: {e}")
            if 'cursor' in locals():
                try:
                    # Make sure to clean up temporary table if it exists
                    cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
                    self.conn.commit()
                except:
                    pass
            return df  # Return original dataframe on error