import pandas as pd
import numpy as np

# Đọc file CSV đã xử lý
input_file = 'data/processed_output.csv'
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error reading {input_file}: {e}")
    exit()

# Các cột chất ô nhiễm
pollutants = ['pm2.5', 'pm10', 'so2', 'no2', 'co', 'o3']

# Điền giá trị thiếu bằng trung vị
for col in pollutants:
    if col in df.columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    else:
        print(f"Column {col} not found in the CSV.")

# Bảng breakpoint AQI theo US EPA (nồng độ: μg/m³ cho pm2.5, pm10, so2, no2; mg/m³ cho co; ppb cho o3)
breakpoints = {
    'pm2.5': [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ],
    'pm10': [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400),
        (505, 604, 401, 500)
    ],
    'so2': [
        (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
        (186, 304, 151, 200), (305, 604, 201, 300), (605, 804, 301, 400),
        (805, 1004, 401, 500)
    ],
    'no2': [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400),
        (1650, 2049, 401, 500)
    ],
    'co': [
        (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500)
    ],
    'o3': [
        (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
        (86, 105, 151, 200), (106, 200, 201, 300), (201, 300, 301, 400),
        (301, 400, 401, 500)
    ]
}

# Hàm tính AQI cho một chất ô nhiễm
def calculate_aqi(concentration, pollutant):
    # Làm tròn nồng độ theo quy tắc EPA
    if pollutant in ['pm2.5', 'pm10']:
        concentration = round(concentration, 1)
    elif pollutant == 'co':
        concentration = round(concentration, 1)
    else:
        concentration = int(concentration)
    
    # Tìm breakpoint phù hợp
    for c_low, c_high, i_low, i_high in breakpoints[pollutant]:
        if c_low <= concentration <= c_high:
            # Công thức AQI
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi)
    return np.nan  # Trả về NaN nếu nồng độ ngoài phạm vi

# Tính AQI cho mỗi bản ghi
aqi_values = []
for idx, row in df.iterrows():
    aqi_per_pollutant = []
    for pollutant in pollutants:
        if pollutant in df.columns:
            aqi = calculate_aqi(row[pollutant], pollutant)
            aqi_per_pollutant.append(aqi)
    # AQI cuối cùng là giá trị tối đa
    aqi_values.append(max(aqi_per_pollutant) if aqi_per_pollutant else np.nan)

# Thêm cột AQI vào DataFrame
df['AQI'] = aqi_values

# Lưu file kết quả
output_file = 'aqi_output.csv'
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"AQI calculated and saved as {output_file}")