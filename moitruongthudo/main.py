import requests
import json
import os
from datetime import datetime

# Các ID và địa điểm tương ứng
locations = {
    14: "số 46 lưu quang vũ",
    15: "minh khai bắc từ liêm",
    31: "đào duy từ",
    38: "an khánh",
    39: "vân hà"
}

# URL theo loại dữ liệu
data_types = {
    "stat": "https://moitruongthudo.vn/public/dailystat/{}",
    "aqi": "https://moitruongthudo.vn/public/dailyaqi/{}"
}

# File lưu trữ dữ liệu
base_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(base_dir, "output.json")
log_file = os.path.join(base_dir, "crawl_log.txt")

def log(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")
        
# Load dữ liệu cũ nếu có
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)
else:
    output_data = {}

# Hàm kiểm tra dữ liệu đã tồn tại (dựa trên 'time')
def data_exists(data_list, new_item):
    return any(d['time'] == new_item['time'] for d in data_list)

# Bắt đầu crawl dữ liệu
for location_id in locations:
    loc_id_str = str(location_id)

    if loc_id_str not in output_data:
        output_data[loc_id_str] = {}

    for dtype, url_template in data_types.items():
        url = url_template.format(location_id)

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if dtype not in output_data[loc_id_str]:
                output_data[loc_id_str][dtype] = {}

            new_records_count = 0  # Đếm số bản ghi mới được thêm

            # Dữ liệu có thể gồm nhiều key như "PM2.5", "PM10",...
            for key in data:
                if key not in output_data[loc_id_str][dtype]:
                    output_data[loc_id_str][dtype][key] = []
                for item in data[key]:
                    if not data_exists(output_data[loc_id_str][dtype][key], item):
                        output_data[loc_id_str][dtype][key].append(item)
                        new_records_count += 1

            log(f"[✓] {dtype.upper()} - ID {location_id} ({locations[location_id]}): +{new_records_count} bản ghi mới")

        except Exception as e:
            log(f"[!] Lỗi khi crawl {dtype} từ {url}: {e}")

# Lưu kết quả
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

log("[✔] Đã lưu dữ liệu phân cấp vào output.json")
log("===== Kết thúc crawl =====\n")