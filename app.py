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
