import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

# Đọc dữ liệu
data_path = Path(__file__).parents[1] / "weather-dataset" / "result" / "hanoiweather_all.csv"
df = pd.read_csv(data_path)

# Tiền xử lý
df["wind_speed"] = df["wind_speed"].str.replace(" km/h", "").astype(float)
df["humidity"] = df["humidity"].str.replace("%", "").astype(float)
df["temperature"] = df["temperature"].str.replace("°", "").astype(float)

# Kiểm tra dữ liệu thiếu
if df.isnull().any().any():
    print("⚠️ Có giá trị thiếu trong dữ liệu. Hãy kiểm tra lại.")
    print(df.isnull().sum())
    df = df.dropna()

# One-hot encoding cho cột phân loại
df = pd.get_dummies(df, columns=["weather_icon", "location"])

# Huấn luyện mô hình KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
df["cluster"] = kmeans.fit_predict(df.drop(["timestamp"], axis=1))

# In kết quả
print(df[["timestamp", "cluster"]])

# Trực quan hóa với PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(df.drop(["timestamp", "cluster"], axis=1))

plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=df["cluster"], cmap="viridis", s=10)
plt.title("KMeans Clustering of Weather Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# Lưu kết quả ra CSV
output_path = Path(__file__).parents[1] / "weather-dataset" / "result" / "hanoiweather_clustered.csv"
df[["timestamp", "cluster"]].to_csv(output_path, index=False)
print(f"✅ Kết quả đã được lưu tại: {output_path}")

# Phân tích đặc trưng trung bình của từng cụm
features = ["temperature", "humidity", "wind_speed"]
cluster_summary = df.groupby("cluster")[features].mean()
print("\n📊 Đặc trưng trung bình của từng cụm:")
print(cluster_summary)

# Gắn nhãn mô tả cho từng cụm
def label_cluster(row):
    if row['cluster'] == 0:
        return "Cụm 0: Nóng - Khô"
    elif row['cluster'] == 1:
        return "Cụm 1: Mát - Ẩm"
    elif row['cluster'] == 2:
        return "Cụm 2: Gió mạnh - Vừa ẩm"

df["cluster_label"] = df.apply(label_cluster, axis=1)

# Thống kê số lượng mỗi cụm
print("\n📈 Số lượng bản ghi trong mỗi cụm:")
print(df["cluster_label"].value_counts())
