import os
import time
import requests
import pandas as pd
import schedule
from datetime import datetime, timedelta, timezone

# —————————————————————————————————————————————
# Cấu hình chung
API_KEY = "e9e7d12f94d484659bdd7ed134c2c1841b9e79e56cab42c57082bc0936f650b5"
HEADERS = {"X-API-Key": API_KEY}
PARAMETERS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# Danh sách LOCATION_ID của 4 trạm tại Hà Nội
LOCATION_IDS = [2161292, 2161296, 2161306, 2161290]

SENSORS_URL_TPL = "https://api.openaq.org/v3/locations/{location_id}/sensors"
MEAS_TPL = "https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
LOCATION_URL_TPL = "https://api.openaq.org/v3/locations/{location_id}"

# Format ISO 8601 với UTC
ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Cache để lưu thông tin location
location_cache = {}

# Biến để theo dõi tháng đã crawl
last_month = None

def should_run_monthly():
    """Kiểm tra xem có nên chạy crawl theo tháng không"""
    global last_month
    now = datetime.now()
    current_month = (now.year, now.month)
    
    # Nếu là ngày đầu tiên của tháng và chưa crawl tháng này
    if now.day == 1 and current_month != last_month:
        last_month = current_month
        return True
    return False

def check_and_run_monthly():
    """Kiểm tra và chạy crawl theo tháng nếu cần"""
    if should_run_monthly():
        try:
            fetch_air_quality("monthly")
        except Exception as e:
            print(f"Monthly fetch lỗi: {e}")

def get_location_info(location_id):
    """Lấy thông tin chi tiết về location từ API"""
    if location_id in location_cache:
        return location_cache[location_id]
        
    try:
        r = requests.get(
            LOCATION_URL_TPL.format(location_id=location_id),
            headers=HEADERS
        )
        r.raise_for_status()
        location_data = r.json().get("results", [{}])[0]
        
        # Lưu vào cache
        location_cache[location_id] = {
            "location_id": location_id,
            "location_name": location_data.get("name"),
            "timezone": location_data.get("timezone"),
            "latitude": location_data.get("coordinates", {}).get("latitude"),
            "longitude": location_data.get("coordinates", {}).get("longitude"),
            "country_iso": location_data.get("country", {}).get("code"),
            "isMobile": location_data.get("isMobile"),
            "isMonitor": location_data.get("isMonitor"),
            "owner_name": location_data.get("owner", {}).get("name"),
            "provider": location_data.get("provider", {}).get("name")
        }
        return location_cache[location_id]
    except Exception as e:
        print(f"Lỗi khi lấy thông tin location {location_id}: {e}")
        return None

def compute_range(period: str):
    """Trả về tuple (date_from, date_to) cho khoảng daily"""
    now = datetime.now(timezone.utc)

    if period == "daily":
        today = datetime(now.year, now.month, now.day)
        start = today - timedelta(days=1)
        end = today
    else:
        raise ValueError(f"Unknown period: {period}")

    return start.strftime(ISO_FMT), end.strftime(ISO_FMT)

def fetch_air_quality(period: str):
    """Crawl dữ liệu measurements cho khoảng period từ các LOCATION_ID tại Hà Nội"""
    df_from, df_to = compute_range(period)
    print(f"[{datetime.now()}] Crawling ({period}) from {df_from} to {df_to}…")

    all_recs = []
    for location_id in LOCATION_IDS:
        print(f"Processing location ID {location_id} (Hà Nội)...")

        # Lấy thông tin location
        location_info = get_location_info(location_id)
        if not location_info:
            print(f"→ Không thể lấy thông tin location {location_id}")
            continue

        # 1) Lấy sensors cho location
        try:
            r = requests.get(
                SENSORS_URL_TPL.format(location_id=location_id),
                headers=HEADERS,
                params={"limit": 1000, "parameters": ",".join(PARAMETERS)}
            )
            r.raise_for_status()
            sensors = [
                s for s in r.json().get("results", [])
                if s["parameter"]["name"].lower() in [p.lower() for p in PARAMETERS]
            ]
            if not sensors:
                print(f"→ Location {location_id}: Không tìm thấy sensor phù hợp.")
                continue

            # Log các cảm biến tìm thấy
            sensor_params = [s["parameter"]["name"] for s in sensors]
            print(f"→ Location {location_id}: Found sensors for {sensor_params}")

            # 2) Thu thập measurements phân trang
            for s in sensors:
                sid = s["id"]
                param = s["parameter"]["name"]
                page = 1
                while True:
                    try:
                        resp = requests.get(
                            MEAS_TPL.format(sensor_id=sid),
                            headers=HEADERS,
                            params={
                                "datetime_from": df_from,
                                "datetime_to": df_to,
                                "limit": 100,
                                "page": page,
                            },
                        )
                        resp.raise_for_status()
                        batch = resp.json().get("results", [])
                        if not batch:
                            break
                            
                        # Thêm thông tin location vào mỗi record
                        for record in batch:
                            record.update(location_info)
                            
                        all_recs.extend(batch)
                        if len(batch) < 100:
                            break
                        page += 1
                    except Exception as e:
                        print(f"→ Lỗi khi lấy measurements cho sensor {sid} (page {page}): {e}")
                        break

        except Exception as e:
            print(f"→ Lỗi khi lấy sensors cho location {location_id}: {e}")
            continue

    if not all_recs:
        print("→ Không có dữ liệu mới.")
        return

    # 3) Chuẩn hoá và lưu
    df = pd.json_normalize(all_recs)
    if df.empty:
        print("→ Dataframe rỗng sau khi chuẩn hoá.")
        return

    # Định dạng lại các cột theo yêu cầu
    df["datetimeUtc"] = pd.to_datetime(df["period.datetimeFrom.utc"])
    df["datetimeLocal"] = pd.to_datetime(df["coverage.datetimeFrom.local"])
    
    # Chọn và sắp xếp các cột theo thứ tự yêu cầu
    desired_columns = [
        "location_id",              # location_id
        "location_name",            # location_name
        "parameter.name",           # parameter
        "value",                    # value
        "parameter.units",          # unit
        "datetimeUtc",              # datetimeUtc
        "datetimeLocal",            # datetimeLocal
        "timezone",                 # timezone
        "latitude",                 # latitude
        "longitude",                # longitude
        "country_iso",              # country_iso
        "isMobile",                 # isMobile
        "isMonitor",                # isMonitor
        "owner_name",               # owner_name
        "provider"                  # provider
    ]

    # Đảm bảo tất cả các cột cần thiết tồn tại
    for col in desired_columns:
        if col not in df.columns and col not in ["datetimeUtc", "datetimeLocal"]:
            df[col] = None

    # Chọn và đổi tên cột
    df = df[desired_columns].rename(columns={
        "parameter.name": "parameter",
        "value": "value",
        "parameter.units": "unit"
    })

    # Tạo thư mục data nếu chưa tồn tại
    os.makedirs("data", exist_ok=True)

    # Lưu vào file CSV tổng hợp theo period
    fname = f"data/air_quality_hanoi_{period}.csv"
    
    try:
        # Nếu file chưa tồn tại, tạo mới với header
        if not os.path.exists(fname):
            df.to_csv(fname, index=False)
        else:
            # Nếu file đã tồn tại, append data mới vào cuối file
            df.to_csv(fname, mode='a', header=False, index=False)
        print(f"✅ Saved {len(df)} records to {fname}")
    except Exception as e:
        print(f"Lỗi khi lưu file {fname}: {e}")

# —————————————————————————————————————————————
# Lên lịch
schedule.every().day.at("00:05").do(lambda: fetch_air_quality("daily"))

# Chạy ngay lần đầu
try:
    fetch_air_quality("daily")
except Exception as e:
    print(f"Initial fetch daily lỗi:", e)

# Vòng lặp chờ schedule
while True:
    schedule.run_pending()
    time.sleep(1)