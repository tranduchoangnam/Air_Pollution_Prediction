import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
input_file = 'data/aqi_output.csv'
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error reading {input_file}: {e}")
    exit()

# Các cột cần trực quan hóa
columns = ['pm2.5', 'pm10', 'so2', 'no2', 'co', 'o3', 'AQI']

# Tạo lưới histogram
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns, 1):
    if col in df.columns:
        plt.subplot(3, 3, i)
        plt.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Phân phối {col.upper()}')
        plt.xlabel(col.upper())
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    else:
        print(f"Column {col} not found in the CSV.")

# Điều chỉnh bố cục để tránh chồng chéo
plt.tight_layout()

# Lưu biểu đồ
output_file = 'data/combined_histograms.png'
plt.savefig(output_file)
plt.close()
print(f"Combined histograms saved as {output_file}")