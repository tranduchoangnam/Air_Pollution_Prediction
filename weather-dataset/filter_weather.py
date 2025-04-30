import pandas as pd

# Đọc file CSV
df = pd.read_csv("result/hanoiweather.csv")

# Chọn và đổi tên cột
df = df[['datetime', 'icon', 'windspeed', 'humidity', 'temp']]
df = df.rename(columns={
    'datetime': 'timestamp',
    'icon': 'weather_icon',
    'windspeed': 'wind_speed',
    'humidity': 'humidity',
    'temp': 'temperature'
})

# Chuẩn hóa timestamp
def standardize_timestamp(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.to_datetime(ts + 'T00:00:00+07:00', utc=True)

df['timestamp'] = df['timestamp'].apply(standardize_timestamp)

# Chuyển về múi giờ Việt Nam
df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Ho_Chi_Minh')

# Chuyển sang chuỗi ISO 8601 với milliseconds và múi giờ
df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat())  # Đảm bảo định dạng như ISO 8601 với milliseconds và múi giờ

# Thêm cột location
df['location'] = 'Hà Nội'

# ✅ Xử lý giá trị thiếu
df['humidity'] = df['humidity'].fillna(df['humidity'].mean())
df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
df['weather_icon'] = df['weather_icon'].fillna(df['weather_icon'].mode()[0])

# ✅ Thêm đơn vị vào các cột tương ứng
df['wind_speed'] = df['wind_speed'].astype(float).round(1).astype(str) + ' km/h'
df['humidity'] = df['humidity'].astype(int).astype(str) + '%'
df['temperature'] = df['temperature'].astype(float).round(1).astype(str) + '°'

# Ghi ra file CSV mới
df.to_csv("result/hanoiweather_filtered.csv", index=False)
print("✅ Đã chuẩn hóa và xử lý dữ liệu thiếu. Đã lưu file mới.")

# Kiểm tra dữ liệu thiếu
print("\n🔍 Dữ liệu thiếu sau khi xử lý:")
print(df.isnull().sum())

# Mô tả dữ liệu
print("\n📊 Thống kê mô tả:")
print(df.describe())
