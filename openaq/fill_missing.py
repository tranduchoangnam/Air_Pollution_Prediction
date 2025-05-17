import pandas as pd
import numpy as np

# Đọc file CSV đã xử lý
input_file = 'data/processed_output.csv'
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error reading {input_file}: {e}")
    exit()

# Danh sách các cột chất gây ô nhiễm
pollutant_columns = ['pm2.5', 'pm10', 'so2', 'no2', 'co', 'o3']

# Kiểm tra sự tồn tại của các cột chất gây ô nhiễm
missing_cols = [col for col in pollutant_columns if col not in df.columns]
if missing_cols:
    print(f"Warning: Columns {missing_cols} not found in the CSV.")
    pollutant_columns = [col for col in pollutant_columns if col in df.columns]

# Điền giá trị thiếu bằng trung vị
for col in pollutant_columns:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

# Lưu file CSV với giá trị đã điền
output_file = 'data/filled_output.csv'
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Filled CSV saved as {output_file}")
