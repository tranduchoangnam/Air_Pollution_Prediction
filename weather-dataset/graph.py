import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
df = pd.read_csv('result/hanoiweather_all.csv')

# Chuyển đổi timestamp về kiểu datetime (tự động nhận diện định dạng ISO 8601)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Làm tròn timestamp về giây
df['timestamp'] = df['timestamp'].dt.round('s')

# Chuyển đổi các giá trị wind_speed, humidity, temperature thành kiểu số học (float)
df['wind_speed'] = df['wind_speed'].str.replace(' km/h', '').astype(float)
df['humidity'] = df['humidity'].str.replace('%', '').astype(float)

# Loại bỏ ký tự '°C' hoặc '°' khỏi cột temperature trước khi chuyển đổi sang float
df['temperature'] = df['temperature'].str.replace('°C', '', regex=False).str.replace('°', '', regex=False).astype(float)

# Kiểm tra dữ liệu thiếu
print("\n🔍 Kiểm tra dữ liệu thiếu:")
print(df.isnull().sum())

# Vẽ biểu đồ về nhiệt độ theo thời gian
plt.figure(figsize=(14,6))
plt.plot(df['timestamp'], df['temperature'], label='Temperature (°C)', color='orange')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Temperature over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Vẽ biểu đồ về độ ẩm theo thời gian
plt.figure(figsize=(14,6))
plt.plot(df['timestamp'], df['humidity'], label='Humidity (%)', color='blue')
plt.xlabel('Time')
plt.ylabel('Humidity (%)')
plt.title('Humidity over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Vẽ biểu đồ về tốc độ gió theo thời gian
plt.figure(figsize=(14,6))
plt.plot(df['timestamp'], df['wind_speed'], label='Wind Speed (km/h)', color='green')
plt.xlabel('Time')
plt.ylabel('Wind Speed (km/h)')
plt.title('Wind Speed over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Phân tích sự phân bố của các biểu tượng thời tiết
plt.figure(figsize=(10,6))
sns.countplot(y='weather_icon', data=df, order=df['weather_icon'].value_counts().index)
plt.title('Weather Icon Distribution')
plt.xlabel('Count')
plt.ylabel('Weather Icon')
plt.tight_layout()
plt.show()
