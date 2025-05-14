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
import plotly.express as px
import plotly.graph_objects as go

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
WHERE state = 'Hanoi'
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

# Kiểm tra và hiển thị dữ liệu cho từng chỉ số
for col in ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']:
    if col in df.columns:
        # Lấy dữ liệu và sắp xếp theo thời gian
        data = df[['timestamp', col]].dropna().sort_values('timestamp')
        if not data.empty:
            st.write(f"Biểu đồ {col}")
            # Vẽ biểu đồ tương tác với Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data[col],
                mode='lines+markers',
                name=col,
                hovertemplate="<b>Thời gian:</b> %{x}<br>" +
                            "<b>Giá trị:</b> %{y}<br>" +
                            "<extra></extra>"
            ))
            
            fig.update_layout(
                title=f'Biến động {col} theo thời gian',
                xaxis_title='Thời gian',
                yaxis_title=col,
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Không có dữ liệu cho chỉ số {col}")
    else:
        st.warning(f"Không tìm thấy cột {col} trong dữ liệu")

# Thời tiết & nhiệt độ
st.subheader("Thời tiết và nhiệt độ")
col1, col2, col3 = st.columns(3)
with col1:
    fig_temp = px.line(df, x='timestamp', y='temperature', 
                      title='Nhiệt độ',
                      labels={'timestamp': 'Thời gian', 'temperature': 'Nhiệt độ (°C)'})
    st.plotly_chart(fig_temp, use_container_width=True)
with col2:
    fig_humid = px.line(df, x='timestamp', y='humidity',
                       title='Độ ẩm',
                       labels={'timestamp': 'Thời gian', 'humidity': 'Độ ẩm (%)'})
    st.plotly_chart(fig_humid, use_container_width=True)
with col3:
    fig_press = px.line(df, x='timestamp', y='pressure',
                       title='Áp suất',
                       labels={'timestamp': 'Thời gian', 'pressure': 'Áp suất (hPa)'})
    st.plotly_chart(fig_press, use_container_width=True)

# Wind direction
st.subheader("Tốc độ và hướng gió")
col1, col2 = st.columns(2)
with col1:
    fig_wind = px.line(df, x='timestamp', y='wind_speed',
                      title='Tốc độ gió',
                      labels={'timestamp': 'Thời gian', 'wind_speed': 'Tốc độ gió (m/s)'})
    st.plotly_chart(fig_wind, use_container_width=True)
with col2:
    fig_dir = px.line(df, x='timestamp', y='wind_direction',
                     title='Hướng gió',
                     labels={'timestamp': 'Thời gian', 'wind_direction': 'Hướng gió (độ)'})
    st.plotly_chart(fig_dir, use_container_width=True)

# Phần dự đoán
st.subheader("Dự đoán Chất lượng Không khí")

# Chọn chỉ số cần dự đoán
target_col = st.selectbox("Chọn chỉ số cần dự đoán:", ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3'])

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
    
    # Hiển thị biểu đồ so sánh với Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Dự đoán',
        hovertemplate="<b>Giá trị thực tế:</b> %{x}<br>" +
                     "<b>Giá trị dự đoán:</b> %{y}<br>" +
                     "<extra></extra>"
    ))
    
    # Thêm đường chéo
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Đường chéo',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        title='So sánh giá trị thực tế và dự đoán',
        xaxis_title='Giá trị thực tế',
        yaxis_title='Giá trị dự đoán',
        hovermode='closest',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("Ứng dụng demo Streamlit - 📍 Hà Nội")
