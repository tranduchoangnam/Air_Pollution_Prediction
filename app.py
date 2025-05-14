import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Kết nối database
def get_db_connection():
    db_url = 'postgresql://tsdbadmin:lnmfese700b796cn@gejinnsvx3.aqgqm1fn3z.tsdb.cloud.timescale.com:35582/tsdb?sslmode=require'
    return create_engine(db_url)

# Đọc dữ liệu từ bảng merge
engine = get_db_connection()
query = """
SELECT timestamp, city, state, country, 
       "NO2", "PM10", "PM2.5", "CO", "SO2", "O3",
       temperature, pressure, humidity,
       wind_speed, wind_direction, weather_icon
FROM merge
WHERE city = 'Hanoi'
ORDER BY timestamp DESC
"""
df = pd.read_sql(query, engine)

# Tiêu đề chính
st.title("Giám sát và Dự đoán Chất lượng Không khí Hà Nội")

# Hiển thị bảng dữ liệu mới nhất
st.subheader("Dữ liệu mới nhất")
latest_data = df.sort_values(by='timestamp', ascending=False).head(1)
st.write(latest_data)

# Biểu đồ thời gian
st.subheader("Biến động các chỉ số theo thời gian")

cols_to_plot = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
for col in cols_to_plot:
    st.line_chart(df.set_index('timestamp')[col])

# Thời tiết & nhiệt độ
st.subheader("Thời tiết và nhiệt độ")
col1, col2, col3 = st.columns(3)
with col1:
    st.line_chart(df.set_index('timestamp')['temperature'])
with col2:
    st.line_chart(df.set_index('timestamp')['humidity'])
with col3:
    st.line_chart(df.set_index('timestamp')['pressure'])

# Wind direction
st.subheader("Tốc độ và hướng gió")
col1, col2 = st.columns(2)
with col1:
    st.line_chart(df.set_index('timestamp')['wind_speed'])
with col2:
    st.line_chart(df.set_index('timestamp')['wind_direction'])

# Phần dự đoán
st.subheader("Dự đoán Chất lượng Không khí")

# Chọn chỉ số cần dự đoán
target_col = st.selectbox("Chọn chỉ số cần dự đoán:", cols_to_plot)

# Chọn mô hình
model_type = st.selectbox("Chọn mô hình:", ["Linear Regression", "Random Forest"])

# Chuẩn bị dữ liệu
def prepare_data(df, target_col):
    # Tạo các đặc trưng từ timestamp
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Chọn các đặc trưng
    features = ['hour', 'day', 'month', 'day_of_week', 
                'temperature', 'pressure', 'humidity', 
                'wind_speed']
    
    X = df[features]
    y = df[target_col]
    
    return X, y

# Huấn luyện và dự đoán
if st.button("Dự đoán"):
    X, y = prepare_data(df, target_col)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Huấn luyện mô hình
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test_scaled)
    
    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R2 Score: {r2:.2f}")
    
    # Hiển thị biểu đồ so sánh
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Giá trị thực tế')
    ax.set_ylabel('Giá trị dự đoán')
    ax.set_title('So sánh giá trị thực tế và dự đoán')
    st.pyplot(fig)

# Footer
st.markdown("Ứng dụng demo Streamlit - Duong Van Nhat | 📍 Hà Nội")
