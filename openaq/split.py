import pandas as pd
import numpy as np

# Đọc file CSV tổng hợp
input_file = 'data/air_quality_hanoi_daily.csv'
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error reading {input_file}: {e}")
    exit()

# Bỏ các cột không cần thiết
columns_to_drop = ['unit', 'timezone', 'country_iso', 'isMobile', 'isMonitor', 'owner_name', 'provider']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Kiểm tra sự tồn tại của cột 'parameter' và 'value'
if 'parameter' not in df.columns or 'value' not in df.columns:
    print("Required columns 'parameter' or 'value' not found in the CSV.")
    exit()

# Chuẩn hóa giá trị trong cột 'parameter' (đảm bảo khớp với pm2.5, pm10, ...)
df['parameter'] = df['parameter'].str.lower().replace({
    'pm25': 'pm2.5',
    'pm_25': 'pm2.5',
    'pm10': 'pm10',
    'so2': 'so2',
    'no2': 'no2',
    'co': 'co',
    'o3': 'o3'
})

# Tạo pivot table để chia cột 'parameter' thành các cột pm2.5, pm10, so2, no2, co, o3
pivot_df = df.pivot_table(
    values='value',
    index=[col for col in df.columns if col not in ['parameter', 'value']],
    columns='parameter',
    aggfunc='first'  # Lấy giá trị đầu tiên nếu có trùng lặp
).reset_index()

# Đảm bảo các cột pm2.5, pm10, so2, no2, co, o3 tồn tại, nếu thiếu thì tạo với giá trị NaN
pollutants = ['pm2.5', 'pm10', 'so2', 'no2', 'co', 'o3']
for pollutant in pollutants:
    if pollutant not in pivot_df.columns:
        pivot_df[pollutant] = np.nan

# Lưu file kết quả
output_file = 'data/processed_output.csv'
pivot_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Processed CSV saved as {output_file}")