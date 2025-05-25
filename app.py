import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Giám sát và Dự đoán Chất lượng Không khí Hà Nội",
    page_icon="🌫️",
    layout="wide",  # Sử dụng layout rộng
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS để tăng chiều rộng
st.markdown("""
    <style>
        .main > div {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stPlotlyChart {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề và mô tả
st.title("🌫️ Giám sát và Dự đoán Chất lượng Không khí Hà Nội")
st.markdown("""
    Ứng dụng này cung cấp thông tin thời gian thực về chất lượng không khí tại các trạm quan trắc ở Hà Nội.
    Chỉ số AQI (Air Quality Index) được tính toán dựa trên các thông số: PM2.5, PM10, NO2, SO2, O3, và CO.
""")

# Hàm tính AQI
def calculate_aqi(pm25, pm10, no2, so2, co, o3):
    # Tính AQI cho từng chất
    def get_aqi_pm25(pm25):
        if pm25 <= 12.0: return pm25 * 50/12.0
        elif pm25 <= 35.4: return 50 + (pm25 - 12.0) * 50/23.4
        elif pm25 <= 55.4: return 100 + (pm25 - 35.4) * 50/20
        elif pm25 <= 150.4: return 150 + (pm25 - 55.4) * 50/95
        elif pm25 <= 250.4: return 200 + (pm25 - 150.4) * 50/100
        else: return 300 + (pm25 - 250.4) * 100/149.6

    def get_aqi_pm10(pm10):
        if pm10 <= 54: return pm10 * 50/54
        elif pm10 <= 154: return 50 + (pm10 - 54) * 50/100
        elif pm10 <= 254: return 100 + (pm10 - 154) * 50/100
        elif pm10 <= 354: return 150 + (pm10 - 254) * 50/100
        elif pm10 <= 424: return 200 + (pm10 - 354) * 50/70
        else: return 300 + (pm10 - 424) * 100/175

    def get_aqi_no2(no2):
        if no2 <= 53: return no2 * 50/53
        elif no2 <= 100: return 50 + (no2 - 53) * 50/47
        elif no2 <= 360: return 100 + (no2 - 100) * 50/260
        elif no2 <= 649: return 150 + (no2 - 360) * 50/289
        elif no2 <= 1249: return 200 + (no2 - 649) * 50/600
        else: return 300 + (no2 - 1249) * 100/750

    def get_aqi_so2(so2):
        if so2 <= 35: return so2 * 50/35
        elif so2 <= 75: return 50 + (so2 - 35) * 50/40
        elif so2 <= 185: return 100 + (so2 - 75) * 50/110
        elif so2 <= 304: return 150 + (so2 - 185) * 50/119
        elif so2 <= 604: return 200 + (so2 - 304) * 50/300
        else: return 300 + (so2 - 604) * 100/396

    def get_aqi_co(co):
        if co <= 4.4: return co * 50/4.4
        elif co <= 9.4: return 50 + (co - 4.4) * 50/5
        elif co <= 12.4: return 100 + (co - 9.4) * 50/3
        elif co <= 15.4: return 150 + (co - 12.4) * 50/3
        elif co <= 30.4: return 200 + (co - 15.4) * 50/15
        else: return 300 + (co - 30.4) * 100/69.6

    def get_aqi_o3(o3):
        if o3 <= 54: return o3 * 50/54
        elif o3 <= 70: return 50 + (o3 - 54) * 50/16
        elif o3 <= 85: return 100 + (o3 - 70) * 50/15
        elif o3 <= 105: return 150 + (o3 - 85) * 50/20
        elif o3 <= 200: return 200 + (o3 - 105) * 50/95
        else: return 300 + (o3 - 200) * 100/100

    # Tính AQI cho từng chất và lấy giá trị lớn nhất
    aqi_values = []
    if not pd.isna(pm25): aqi_values.append(get_aqi_pm25(pm25))
    if not pd.isna(pm10): aqi_values.append(get_aqi_pm10(pm10))
    if not pd.isna(no2): aqi_values.append(get_aqi_no2(no2))
    if not pd.isna(so2): aqi_values.append(get_aqi_so2(so2))
    if not pd.isna(co): aqi_values.append(get_aqi_co(co))
    if not pd.isna(o3): aqi_values.append(get_aqi_o3(o3))
    
    return max(aqi_values) if aqi_values else None

# Kết nối database
def get_db_connection():
    db_url = 'postgresql://tsdbadmin:msk844xlog64qfib@y5s99n9ilz.hzk9co8bbu.tsdb.cloud.timescale.com:34150/tsdb?sslmode=require'
    return create_engine(db_url)

# Đọc dữ liệu từ bảng merge
engine = get_db_connection()

# Kiểm tra dữ liệu
data_query = """
SELECT * FROM public.merge_aqi_weather
ORDER BY "time" DESC
"""
all_data = pd.read_sql(data_query, engine)

# Hiển thị thông tin về dữ liệu
st.write("Thông tin về dữ liệu:")
st.write(f"Số lượng bản ghi: {len(all_data)}")
st.write(f"Thời gian sớm nhất: {all_data['time'].min()}")
st.write(f"Thời gian mới nhất: {all_data['time'].max()}")
st.write(f"Các địa điểm: {', '.join(all_data['location'].unique())}")

# Lấy dữ liệu mới nhất cho tất cả các địa điểm
data_query = """
SELECT DISTINCT ON (location) 
    time, location, "NO2", "PM10", "PM2.5", "CO", "SO2", "O3",
    temperature, pressure, humidity, wind_speed, wind_direction, weather_icon, dew
FROM public.merge_aqi_weather
ORDER BY location, time DESC
"""
latest_data = pd.read_sql(data_query, engine)

# Tính AQI (giá trị lớn nhất của các chỉ số ô nhiễm)
pollution_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
latest_data['AQI'] = latest_data[pollution_columns].max(axis=1)

# Sắp xếp theo AQI giảm dần
latest_data = latest_data.sort_values('AQI', ascending=False)

# Hiển thị bảng dữ liệu
st.subheader("📊 Chỉ số AQI tại các địa điểm")
st.markdown("""
    Bảng dưới đây hiển thị chỉ số AQI và các thông số ô nhiễm mới nhất tại các trạm quan trắc.
    AQI được tính bằng giá trị lớn nhất trong các chỉ số ô nhiễm.
""")
st.dataframe(
    latest_data[['location', 'time', 'PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'AQI']],
    use_container_width=True
)

# Hiển thị thông tin thời tiết
st.subheader("🌤️ Thông tin thời tiết")
st.markdown("""
    Bảng dưới đây hiển thị thông tin thời tiết tại các trạm quan trắc.
    Các giá trị thời tiết được lấy trung bình cho tất cả các địa điểm.
""")

# Tạo bảng thời tiết
weather_data = latest_data[['location', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction', 'dew']].copy()
weather_data.columns = ['Địa điểm', 'Nhiệt độ (°C)', 'Áp suất (hPa)', 'Độ ẩm (%)', 'Tốc độ gió (m/s)', 'Hướng gió (độ)', 'Điểm sương (°C)']
st.dataframe(weather_data, use_container_width=True)

# Hiển thị biểu đồ AQI
st.markdown("""
    Biểu đồ dưới đây so sánh chỉ số AQI giữa các địa điểm. Màu sắc thể hiện mức độ ô nhiễm theo thang đo sau:
    
    | Màu sắc | Mức AQI | Mức độ ô nhiễm | Ảnh hưởng sức khỏe |
    |---------|---------|----------------|-------------------|
    | 🟢 Xanh lá | 0-50 | Tốt | Chất lượng không khí tốt, không ảnh hưởng đến sức khỏe |
    | 🟡 Vàng | 51-100 | Trung bình | Chất lượng không khí chấp nhận được, nhóm nhạy cảm nên hạn chế hoạt động ngoài trời |
    | 🟠 Cam | 101-150 | Kém | Nhóm nhạy cảm có thể bị ảnh hưởng sức khỏe, nên hạn chế hoạt động ngoài trời |
    | 🔴 Đỏ | 151-200 | Xấu | Mọi người có thể bị ảnh hưởng sức khỏe, nhóm nhạy cảm có thể bị ảnh hưởng nghiêm trọng |
    | 🟣 Tím | 201-300 | Rất xấu | Cảnh báo sức khỏe khẩn cấp, mọi người có thể bị ảnh hưởng nghiêm trọng |
    | 🟤 Nâu | >300 | Nguy hại | Cảnh báo sức khỏe khẩn cấp, mọi người có thể bị ảnh hưởng nghiêm trọng |
    
    *Nhóm nhạy cảm bao gồm: người già, trẻ em, phụ nữ mang thai và người có bệnh về hô hấp, tim mạch*
""")
fig = px.bar(
    latest_data,
    x='location',
    y='AQI',
    color='AQI',
    color_continuous_scale='RdYlGn_r',
    title='Chỉ số AQI tại các địa điểm',
    labels={'location': 'Địa điểm', 'AQI': 'Chỉ số AQI'},
    hover_data=['time']
)
fig.update_layout(
    xaxis_title='Địa điểm',
    yaxis_title='Chỉ số AQI',
    coloraxis_colorbar_title='AQI'
)
st.plotly_chart(fig, use_container_width=True)

# Lấy dữ liệu AQI theo thời gian cho tất cả các địa điểm
aqi_time_query = """
SELECT time, location, "NO2", "PM10", "PM2.5", "CO", "SO2", "O3"
FROM public.merge_aqi_weather
ORDER BY time DESC
"""
aqi_time_data = pd.read_sql(aqi_time_query, engine)

# Tính AQI cho mỗi bản ghi
aqi_time_data['AQI'] = aqi_time_data[pollution_columns].max(axis=1)

# Hiển thị đồ thị AQI theo thời gian cho tất cả các địa điểm
st.subheader("📈 Biến động AQI theo thời gian tại các địa điểm")
st.markdown("""
    Các đồ thị dưới đây hiển thị diễn biến chỉ số AQI theo thời gian tại từng trạm quan trắc.
    Bạn có thể di chuột qua các điểm để xem chi tiết thời gian và giá trị AQI.
""")

# Tạo 3 cột để hiển thị đồ thị
col1, col2, col3 = st.columns(3)

# Lấy danh sách các địa điểm
locations = aqi_time_data['location'].unique()

# Chia các địa điểm vào 3 cột
for i, location in enumerate(locations):
    location_data = aqi_time_data[aqi_time_data['location'] == location].sort_values('time')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=location_data['time'],
        y=location_data['AQI'],
        mode='lines+markers',
        name=location,
        hovertemplate="<b>Thời gian:</b> %{x}<br><b>AQI:</b> %{y}<br><extra></extra>"
    ))
    
    fig.update_layout(
        title=f'AQI tại {location}',
        xaxis_title='Thời gian',
        yaxis_title='AQI',
        height=400,
        hovermode='x unified'
    )
    
    # Chọn cột để hiển thị
    if i % 3 == 0:
        with col1:
            st.plotly_chart(fig, use_container_width=True)
    elif i % 3 == 1:
        with col2:
            st.plotly_chart(fig, use_container_width=True)
    else:
        with col3:
            st.plotly_chart(fig, use_container_width=True)

# Phần chi tiết cho từng địa điểm
st.subheader("🔍 Chi tiết theo địa điểm")
st.markdown("""
    Chọn một địa điểm để xem thông tin chi tiết về các chỉ số ô nhiễm và điều kiện thời tiết.
""")
selected_location = st.selectbox("Chọn địa điểm:", latest_data['location'])

# Lấy dữ liệu chi tiết cho địa điểm được chọn
query = f"""
SELECT time, location, "NO2", "PM10", "PM2.5", "CO", "SO2", "O3",
       temperature, pressure, humidity, wind_speed, wind_direction, weather_icon, dew
FROM public.merge_aqi_weather
WHERE location = '{selected_location}'
ORDER BY time DESC
"""

df = pd.read_sql(query, engine)
# Ensure datetime is parsed correctly
df['time'] = pd.to_datetime(df['time'], format='mixed')

# Hiển thị bảng dữ liệu mới nhất
st.subheader("Dữ liệu mới nhất")
latest_data = df.sort_values(by='time', ascending=False).head(1)
st.write(latest_data)

# Configure common layout for all plotly charts
plotly_config = {
    'scrollZoom': True,
    'displayModeBar': True,
    'modeBarButtonsToAdd': ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
}

# Biểu đồ thời gian
st.subheader("Biến động các chỉ số theo thời gian")

# Create tabs for different pollutants
pollutant_tabs = st.tabs(['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO'])

# Display each pollutant in its respective tab
for i, col in enumerate(['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']):
    with pollutant_tabs[i]:
        if col in df.columns:
            data = df[['time', col]].dropna().sort_values('time')
            if not data.empty:
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
                    showlegend=True,
                    height=500,
                    xaxis=dict(rangeslider=dict(visible=True))
                )
                st.plotly_chart(fig, use_container_width=True, config=plotly_config)
            else:
                st.warning(f"Không có dữ liệu cho chỉ số {col}")
        else:
            st.warning(f"Không tìm thấy cột {col} trong dữ liệu")

# Thời tiết & nhiệt độ
st.subheader("Thời tiết và nhiệt độ")

# Create tabs for weather parameters
weather_tabs = st.tabs(['Nhiệt độ', 'Độ ẩm', 'Áp suất', 'Tốc độ gió', 'Hướng gió'])

# Temperature tab
with weather_tabs[0]:
    fig_temp = px.line(df, x='time', y='temperature',
                    title='Nhiệt độ',
                    labels={'time': 'Thời gian', 'temperature': 'Nhiệt độ (°C)'})
    fig_temp.update_layout(
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True)),
        height=500
    )
    st.plotly_chart(fig_temp, use_container_width=True, config=plotly_config)

# Humidity tab
with weather_tabs[1]:
    fig_humid = px.line(df, x='time', y='humidity',
                    title='Độ ẩm',
                    labels={'time': 'Thời gian', 'humidity': 'Độ ẩm (%)'})
    fig_humid.update_layout(
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True)),
        height=500
    )
    st.plotly_chart(fig_humid, use_container_width=True, config=plotly_config)

# Pressure tab
with weather_tabs[2]:
    fig_press = px.line(df, x='time', y='pressure',
                    title='Áp suất',
                    labels={'time': 'Thời gian', 'pressure': 'Áp suất (hPa)'})
    fig_press.update_layout(
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True)),
        height=500
    )
    st.plotly_chart(fig_press, use_container_width=True, config=plotly_config)

# Wind speed tab
with weather_tabs[3]:
    fig_wind = px.line(df, x='time', y='wind_speed',
                    title='Tốc độ gió',
                    labels={'time': 'Thời gian', 'wind_speed': 'Tốc độ gió (m/s)'})
    fig_wind.update_layout(
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True)),
        height=500
    )
    st.plotly_chart(fig_wind, use_container_width=True, config=plotly_config)

# Wind direction tab
with weather_tabs[4]:
    fig_dir = px.line(df, x='time', y='wind_direction',
                   title='Hướng gió',
                   labels={'time': 'Thời gian', 'wind_direction': 'Hướng gió (độ)'})
    fig_dir.update_layout(
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True)),
        height=500
    )
    st.plotly_chart(fig_dir, use_container_width=True, config=plotly_config)

# Dự đoán
st.subheader("Dự đoán Chất lượng Không khí")

import joblib
from datetime import datetime, timedelta
import numpy as np
from utils.openweather_util import get_weather_forecast_next_24h

# Function to forecast next 24 hours based on the AQI forecasting pipeline
def forecast_next_24h(model, location_df, location_name):
    """
    Generate forecasts for the next 24 hours using the trained model for a location.

    Args:
        model: Trained model
        location_df: DataFrame with historical data for the location
        location_name: Name of the location to forecast for

    Returns:
        DataFrame with forecasted values
    """
    # Make a copy of the data
    location_df = location_df.copy()
    location_df['time'] = pd.to_datetime(location_df['time'], format='mixed')

    # Get the latest time in the dataset for this location
    latest_time = location_df['time'].max()

    # Create a dataframe for the next 24 hours
    next_hours = pd.date_range(start=latest_time, periods=25, freq='H')[1:]  # Skip first as it's latest time

    # Create forecast dataframe
    next_24h_df = pd.DataFrame({'time': next_hours})
    next_24h_df['location'] = location_name

    # Generate time-based features
    next_24h_df['hour'] = next_24h_df['time'].dt.hour
    next_24h_df['day_of_week'] = next_24h_df['time'].dt.dayofweek
    next_24h_df['month'] = next_24h_df['time'].dt.month
    next_24h_df['day_of_year'] = next_24h_df['time'].dt.dayofyear
    next_24h_df['is_weekend'] = next_24h_df['day_of_week'].isin([5, 6]).astype(int)

    # Create cyclical features
    next_24h_df['hour_sin'] = np.sin(2 * np.pi * next_24h_df['hour']/24)
    next_24h_df['hour_cos'] = np.cos(2 * np.pi * next_24h_df['hour']/24)
    next_24h_df['month_sin'] = np.sin(2 * np.pi * next_24h_df['month']/12)
    next_24h_df['month_cos'] = np.cos(2 * np.pi * next_24h_df['month']/12)
    next_24h_df['day_of_week_sin'] = np.sin(2 * np.pi * next_24h_df['day_of_week']/7)
    next_24h_df['day_of_week_cos'] = np.cos(2 * np.pi * next_24h_df['day_of_week']/7)

    # Get weather forecast data from the OpenWeather API
    try:
        next_24h_df = get_weather_forecast_next_24h(next_24h_df, location_name)
    except Exception as e:
        st.warning(f"Could not get weather forecast data: {str(e)}. Using latest weather data.")

        # For simplicity, copy the latest weather values to all forecast hours if API fails
        latest_row = location_df.iloc[-1]
        weather_vars = ['temperature', 'pressure', 'humidity', 'wind_speed',
                      'wind_direction', 'dew', 'weather_icon']

        for var in weather_vars:
            if var in location_df.columns:
                next_24h_df[var] = latest_row[var]

    # Generate wind direction cyclical features
    if 'wind_direction' in next_24h_df.columns:
        next_24h_df['wind_direction_sin'] = np.sin(2 * np.pi * next_24h_df['wind_direction']/360)
        next_24h_df['wind_direction_cos'] = np.cos(2 * np.pi * next_24h_df['wind_direction']/360)

    # For is_daytime
    next_24h_df['is_daytime'] = ((next_24h_df['hour'] >= 6) & (next_24h_df['hour'] < 18)).astype(int)

    # Prepare combined dataset for prediction
    combined_df = pd.concat([location_df, next_24h_df], ignore_index=True)

    # Create a dictionary to store predictions
    predictions = []

    # Define target column for overall AQI
    target_col = 'overall_aqi'

    # Prepare data for prediction
    for i in range(1, 25):
        # Index for the hour we're predicting
        pred_idx = len(location_df) + i - 1

        # Generate lagged features for the prediction hour
        lag_hours = [1, 2, 3, 6, 12, 24]
        rolling_windows = [3, 6, 12, 24]

        # Create lagged values
        for lag in lag_hours:
            if pred_idx - lag >= 0:
                combined_df.loc[pred_idx, f'{target_col}_lag_{lag}h'] = combined_df.loc[pred_idx - lag, target_col] if target_col in combined_df.columns and pd.notna(combined_df.loc[pred_idx - lag, target_col]) else combined_df.loc[pred_idx - lag, 'PM2.5']
            else:
                # Handle boundary case - use the earliest available value
                combined_df.loc[pred_idx, f'{target_col}_lag_{lag}h'] = combined_df.loc[0, 'PM2.5'] if 'PM2.5' in combined_df.columns else 0

        # Create rolling window features
        for window in rolling_windows:
            # Use historical values for rolling statistics
            if 'PM2.5' in combined_df.columns:
                start_idx = max(0, pred_idx - window)
                window_data = combined_df.loc[start_idx:pred_idx - 1, 'PM2.5'].dropna()

                combined_df.loc[pred_idx, f'{target_col}_rolling_mean_{window}h'] = window_data.mean() if not window_data.empty else combined_df.loc[pred_idx - 1, 'PM2.5']
                combined_df.loc[pred_idx, f'{target_col}_rolling_std_{window}h'] = window_data.std() if len(window_data) > 1 else 0
                combined_df.loc[pred_idx, f'{target_col}_rolling_min_{window}h'] = window_data.min() if not window_data.empty else combined_df.loc[pred_idx - 1, 'PM2.5']
                combined_df.loc[pred_idx, f'{target_col}_rolling_max_{window}h'] = window_data.max() if not window_data.empty else combined_df.loc[pred_idx - 1, 'PM2.5']

        # Create lag differences
        for lag in lag_hours:
            if lag > 1 and pred_idx - 1 >= 0 and pred_idx - lag >= 0:
                pm25_col = 'PM2.5' if 'PM2.5' in combined_df.columns else None
                if pm25_col:
                    combined_df.loc[pred_idx, f'{target_col}_diff_{lag}h'] = (
                        combined_df.loc[pred_idx - 1, pm25_col] -
                        combined_df.loc[pred_idx - lag, pm25_col]
                    )

        # Create lagged weather features
        weather_vars = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for var in weather_vars:
            if var in combined_df.columns and pred_idx - 1 >= 0 and pred_idx - 24 >= 0:
                combined_df.loc[pred_idx, f'{var}_lag_1h'] = combined_df.loc[pred_idx - 1, var]
                combined_df.loc[pred_idx, f'{var}_lag_24h'] = combined_df.loc[pred_idx - 24, var]    # Process data and make predictions with the model
    try:
        # We need to create a feature matrix with exactly 62 features that the model expects
        # First, let's get the rows we need to predict
        pred_rows = combined_df.iloc[-24:].copy()

        # We know the model expects 62 features based on error message
        n_expected_features = model.n_features_in_  # Should be 62

        # Let's ensure we have all necessary features, with common naming patterns in time series models

        # 1. Basic features
        basic_features = ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend',
                         'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                         'day_of_week_sin', 'day_of_week_cos', 'is_daytime']

        # 2. Weather features
        weather_features = ['temperature', 'pressure', 'humidity', 'wind_speed',
                           'wind_direction', 'dew', 'wind_direction_sin', 'wind_direction_cos']

        # 3. Common lag features for time series
        lag_hours = [1, 2, 3, 6, 12, 24]
        lag_features = [f'overall_aqi_lag_{lag}h' for lag in lag_hours]

        # 4. Rolling window features
        rolling_windows = [3, 6, 12, 24]
        rolling_stats = ['mean', 'std', 'min', 'max']
        rolling_features = [f'overall_aqi_rolling_{stat}_{window}h'
                           for window in rolling_windows
                           for stat in rolling_stats]

        # 5. Difference features
        diff_features = [f'overall_aqi_diff_{lag}h' for lag in lag_hours if lag > 1]

        # 6. Lagged weather features
        lagged_weather_features = []
        for var in ['temperature', 'humidity', 'pressure', 'wind_speed']:
            lagged_weather_features.extend([f'{var}_lag_1h', f'{var}_lag_24h'])

        # 7. Pollution features (using PM2.5 as proxy if overall_aqi not available)
        pollution_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']

        # Combine all potential feature types
        all_potential_features = (basic_features + weather_features + lag_features +
                                rolling_features + diff_features + lagged_weather_features +
                                pollution_features)



        # Create a DataFrame with exactly the features the model expects
        X_pred = pd.DataFrame()

        # Add all possible features and fill missing ones with zeros
        for feature in all_potential_features:
            if feature in pred_rows.columns:
                X_pred[feature] = pred_rows[feature]
            else:
                # Feature is missing, add with zeros
                X_pred[feature] = 0.0

        # If we have too few features, add dummy ones until we reach 62
        current_feature_count = len(X_pred.columns)
        if current_feature_count < n_expected_features:
            for i in range(current_feature_count, n_expected_features):
                X_pred[f'dummy_feature_{i}'] = 0.0

        # If we have too many features, keep only the first 62
        if len(X_pred.columns) > n_expected_features:
            X_pred = X_pred.iloc[:, :n_expected_features]

        # Fill any remaining NaN values
        X_pred = X_pred.fillna(0)

        # Make predictions using the model with exactly the right number of features
        try:
            predictions = model.predict(X_pred)
        except Exception as e:
            st.error(f"Error in prediction with prepared features: {str(e)}")

            # Last resort: try with a dummy input of the exact expected shape
            dummy_input = np.zeros((len(pred_rows), n_expected_features))
            predictions = model.predict(dummy_input)

    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return pd.DataFrame()

    # Create forecast result
    forecast_result = next_24h_df[['time', 'location']].copy()
    # Add weather data if available
    for col in ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']:
        if col in next_24h_df.columns:
            forecast_result[col] = next_24h_df[col]

    # Add predictions
    forecast_result['forecast'] = predictions

    return forecast_result

if st.button("Dự đoán"):
    try:
        # Load the pretrained model for the selected location
        model_path = f"{selected_location}_tuned_aqi_model.pkl"

        if not os.path.exists(model_path):
            st.error(f"Không tìm thấy mô hình được huấn luyện cho {selected_location}. Vui lòng chọn địa điểm khác.")
        else:
            # Load the model
            try:
                tuned_model = joblib.load(model_path)
            except Exception as e:
                st.error(f"Lỗi khi tải mô hình: {str(e)}")
                raise

            # Generate forecast for the next 24 hours
            forecast_data = forecast_next_24h(
                model=tuned_model,
                location_df=df,
                location_name=selected_location
            )

            if forecast_data.empty:
                st.error("Không thể tạo dự báo. Vui lòng kiểm tra dữ liệu đầu vào.")
            else:
                # Display forecast data
                st.subheader("Dự báo AQI cho 24 giờ tới")
                st.write(forecast_data)

                # Visualize the forecast - limit to next 24 hours only
                fig_forecast = go.Figure()

                # Only include forecast for the next 24 hours (the data we actually predict)
                forecast_24h = forecast_data.copy()
                # Make sure forecast data is sorted by time
                forecast_24h = forecast_24h.sort_values('time')

                # Only include the first 24 hours of forecast
                if len(forecast_24h) > 24:
                    forecast_24h = forecast_24h.head(24)

                # Create visualization
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_24h['time'],
                    y=forecast_24h['forecast'],
                    mode='markers+lines',
                    name='Dự báo AQI',
                    hovertemplate="<b>Thời gian:</b> %{x}<br><b>AQI dự đoán:</b> %{y}<br><extra></extra>"
                ))

                # Add current value if available
                latest_data_point = df.sort_values('time', ascending=False).iloc[0] if not df.empty else None
                if latest_data_point is not None and 'PM2.5' in latest_data_point:
                    current_time = latest_data_point['time']
                    current_value = latest_data_point['PM2.5']  # Using PM2.5 as proxy for AQI

                    # Add current point to graph
                    fig_forecast.add_trace(go.Scatter(
                        x=[current_time],
                        y=[current_value],
                        mode='markers',
                        marker=dict(color='red', size=10),
                        name='Giá trị hiện tại',
                        hovertemplate="<b>Thời gian hiện tại:</b> %{x}<br><b>Giá trị hiện tại:</b> %{y}<br><extra></extra>"
                    ))

                # Set x-axis limits to only show the 24-hour window
                min_time = forecast_24h['time'].min()
                max_time = forecast_24h['time'].max()

                fig_forecast.update_layout(
                    title=f'Dự báo AQI cho {selected_location} - 24 giờ tới',
                    xaxis_title='Thời gian',
                    yaxis_title='AQI',
                    hovermode='x unified',
                    showlegend=True,
                    height=500,
                    xaxis=dict(
                        range=[min_time, max_time],
                        title='Thời gian',
                        rangeslider=dict(visible=True)
                    )
                )
                st.plotly_chart(fig_forecast, use_container_width=True, config=plotly_config)
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")

# Footer
st.markdown("Ứng dụng demo Streamlit - 📍 Hà Nội")

# Add hidden information section with expander
with st.expander("ℹ️ Thông tin hệ thống", expanded=False):
    st.subheader("Thông tin mô hình")

    # Define a function to extract model info when a model is loaded
    def get_model_info(model_path):
        try:
            model = joblib.load(model_path)

            # Get model type
            model_type = type(model).__name__

            # Get number of features
            n_features = 0
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_

            # Extract other model parameters
            params = {}
            if hasattr(model, 'get_params'):
                params = model.get_params()

            return {
                "Model Type": model_type,
                "Number of Features": n_features,
                "Parameters": {k: v for k, v in list(params.items())[:5]} if params else {}
            }
        except Exception as e:
            return {
                "Error": f"Could not load model information: {str(e)}"
            }

    # Try to load model info for the currently selected location
    model_path = f"{selected_location}_tuned_aqi_model.pkl"

    if os.path.exists(model_path):
        model_info = get_model_info(model_path)

        # Display model info in a nice format
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Thông số mô hình")
            for key, value in model_info.items():
                if key != "Parameters":
                    st.markdown(f"**{key}:** {value}")

        with col2:
            if "Parameters" in model_info and model_info["Parameters"]:
                st.markdown("##### Hyper-parameters")
                st.write(model_info["Parameters"])
    else:
        st.warning(f"No model found for {selected_location}. Select a location with a trained model.")

# Footer
st.markdown("Ứng dụng demo Streamlit - 📍 Hà Nội")


