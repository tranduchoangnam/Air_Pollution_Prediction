import streamlit as st
import datetime
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
locations = pd.read_sql("SELECT DISTINCT location FROM merge_aqi_weather", engine)
selected_location = st.selectbox("Chọn địa điểm:", locations['location'])
query = f"""
SELECT time, location, "NO2", "PM10", "PM2.5", "CO", "SO2", "O3",
       temperature, pressure, humidity, wind_speed, wind_direction, weather_icon, dew
FROM merge_aqi_weather
WHERE location = '{selected_location}'
ORDER BY time DESC
"""

df = pd.read_sql(query, engine)

# Tiêu đề chính
st.title("Giám sát và Dự đoán Chất lượng Không khí Hà Nội")

# Hiển thị bảng dữ liệu mới nhất
st.subheader("Dữ liệu mới nhất")
latest_data = df.sort_values(by='time', ascending=False).head(1)
st.write(latest_data)

# Biểu đồ thời gian
st.subheader("Biến động các chỉ số theo thời gian")

for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']:
    if col in df.columns:
        data = df[['time', col]].dropna().sort_values('time')
        if not data.empty:
            st.write(f"Biểu đồ {col}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=data[col],
                mode='lines+markers',
                name=col,
                hovertemplate="<b>Thời gian:</b> %{x}<br><b>Giá trị:</b> %{y}<br><extra></extra>"
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
    fig_temp = px.line(df, x='time', y='temperature',
                      title='Nhiệt độ',
                      labels={'time': 'Thời gian', 'temperature': 'Nhiệt độ (°C)'})
    st.plotly_chart(fig_temp, use_container_width=True)
with col2:
    fig_humid = px.line(df, x='time', y='humidity',
                       title='Độ ẩm',
                       labels={'time': 'Thời gian', 'humidity': 'Độ ẩm (%)'})
    st.plotly_chart(fig_humid, use_container_width=True)
with col3:
    fig_press = px.line(df, x='time', y='pressure',
                       title='Áp suất',
                       labels={'time': 'Thời gian', 'pressure': 'Áp suất (hPa)'})
    st.plotly_chart(fig_press, use_container_width=True)

# Tốc độ & hướng gió
st.subheader("Tốc độ và hướng gió")
col1, col2 = st.columns(2)
with col1:
    fig_wind = px.line(df, x='time', y='wind_speed',
                      title='Tốc độ gió',
                      labels={'time': 'Thời gian', 'wind_speed': 'Tốc độ gió (m/s)'})
    st.plotly_chart(fig_wind, use_container_width=True)
with col2:
    fig_dir = px.line(df, x='time', y='wind_direction',
                     title='Hướng gió',
                     labels={'time': 'Thời gian', 'wind_direction': 'Hướng gió (độ)'})
    st.plotly_chart(fig_dir, use_container_width=True)

# Dự đoán
st.subheader("Dự đoán Chất lượng Không khí")
target_col = st.selectbox("Chọn chỉ số cần dự đoán:", ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3'])
model_type = st.selectbox("Chọn mô hình:", ["Linear Regression", "Random Forest"])

# Chuẩn bị dữ liệu
def prepare_data(df, target_col):
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Loại bỏ hàng có NaN trong cột target_col
    df = df.dropna(subset=[target_col])
    
    features = ['hour', 'day', 'month', 'day_of_week', 
                'temperature', 'pressure', 'humidity', 
                'wind_speed']
    X = df[features]
    y = df[target_col]
    return X, y

if st.button("Dự đoán"):
    X, y = prepare_data(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Dự đoán cho các ngày tới
    num_days_to_predict = 7  # Ví dụ: dự đoán cho 7 ngày tới
    last_date = df['time'].max()
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, num_days_to_predict + 1)]
    
    future_data = pd.DataFrame({
        'time': future_dates,
        'hour': [date.hour for date in future_dates],
        'day': [date.day for date in future_dates],
        'month': [date.month for date in future_dates],
        'day_of_week': [date.weekday() for date in future_dates],
        'temperature': [df['temperature'].mean()] * num_days_to_predict,
        'pressure': [df['pressure'].mean()] * num_days_to_predict,
        'humidity': [df['humidity'].mean()] * num_days_to_predict,
        'wind_speed': [df['wind_speed'].mean()] * num_days_to_predict
    })
    
    X_future = future_data[['hour', 'day', 'month', 'day_of_week', 
                             'temperature', 'pressure', 'humidity', 
                             'wind_speed']]
    X_future_scaled = scaler.transform(X_future)
    
    future_predictions = model.predict(X_future_scaled)
    future_data['predicted_value'] = future_predictions
    
    # Lấy dữ liệu mới nhất cho chỉ số ô nhiễm
    latest_value = df[df['time'] == last_date][target_col].values[0] if not df[df['time'] == last_date].empty else None

    # Vẽ biểu đồ cho dữ liệu mới nhất
    st.subheader("Biểu đồ dữ liệu mới nhất")
    if latest_value is not None:
        latest_data = pd.DataFrame({
            'time': [last_date],
            'predicted_value': [latest_value]
        })
        
        fig_latest = go.Figure()
        fig_latest.add_trace(go.Scatter(
            x=latest_data['time'],
            y=latest_data['predicted_value'],
            mode='markers+lines',
            name='Giá trị mới nhất',
            hovertemplate="<b>Thời gian:</b> %{x}<br><b>Giá trị:</b> %{y}<br><extra></extra>"
        ))
        fig_latest.update_layout(
            title='Dữ liệu mới nhất',
            xaxis_title='Thời gian',
            yaxis_title=target_col,
            hovermode='x unified',
            showlegend=True
        )
        st.plotly_chart(fig_latest, use_container_width=True)

    # Vẽ biểu đồ cho các dự đoán
    st.subheader("Biểu đồ dự đoán cho các ngày tới")
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(
        x=future_data['time'],
        y=future_data['predicted_value'],
        mode='markers+lines',
        name='Dự đoán',
        hovertemplate="<b>Thời gian:</b> %{x}<br><b>Giá trị dự đoán:</b> %{y}<br><extra></extra>"
    ))
    fig_future.update_layout(
        title='Dự đoán cho các ngày tới',
        xaxis_title='Thời gian',
        yaxis_title=target_col,
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # So sánh giá trị thực tế và dự đoán
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Dự đoán',
        hovertemplate="<b>Giá trị thực tế:</b> %{x}<br><b>Giá trị dự đoán:</b> %{y}<br><extra></extra>"
    ))

    # Tính toán min và max
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    # Vẽ đường chéo
    fig_comparison.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Đường chéo',
        line=dict(dash='dash', color='red')
    ))

    # Cập nhật layout cho biểu đồ
    fig_comparison.update_layout(
        title='So sánh giá trị thực tế và dự đoán',
        xaxis_title='Giá trị thực tế',
        yaxis_title='Giá trị dự đoán',
        hovermode='closest',
        showlegend=True
    )

    # Hiển thị biểu đồ
    st.plotly_chart(fig_comparison, use_container_width=True)

# Footer
st.markdown("Ứng dụng demo Streamlit - 📍 Hà Nội")
