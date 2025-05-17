import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import re
import pandas as pd
import sys

# Add the parent directory to sys.path to import TimescaleDBUtil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timescaledb_util import TimescaleDBUtil

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# ==== Hàm kiểm tra bài viết có liên quan Hà Nội không ====
def is_related_to_hanoi(title, content):
    keywords = ["Hà Nội", "ha noi", "TP Hà Nội", "tp. hà nội", "thủ đô", "nội thành", "ngoại thành"]
    text = (title + " " + content).lower()
    return any(keyword.lower() in text for keyword in keywords)

# ==== Hàm kiểm tra từ khóa liên quan đến không khí ====
def is_air_related(title, content):
    keywords = [ "bụi", "bụi mịn", "PM2.5", "khí thải", "khí độc", "cháy", "cháy rừng", 
    "đốt", "đốt rác", "đốt rơm rạ", "ô nhiễm không khí", "khói", "lò đốt", "nhà máy", 
    "không khí", "haze", "smog", "nhiệt độ không khí", "sương mù", "mù khô", "nghẹt thở", "CO2", "NO2"]
    text = (title + " " + content).lower()
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in text]
    return matched_keywords

# ==== Đọc các link đã crawl trước đó từ file JSON ====
def read_existing_links(filename):
    links = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if "link" in item:
                    links.add(item["link"])
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return links

# ==== Crawl VnExpress (20 trang) ====
def crawl_vnexpress(max_pages=20, existing_links=None):
    if existing_links is None:
        existing_links = set()
    base_url = "https://vnexpress.net/chu-de/o-nhiem-moi-truong-6877"
    headers = {"User-Agent": "Mozilla/5.0"}
    data = []

    for i in range(1, max_pages + 1):
        url = base_url if i == 1 else f"{base_url}-p{i}"
        print(f"🌐 VnExpress: {url}")
        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")
            articles = soup.select(".featured-item, .item-news")

            for article in articles:
                a_tag = article.select_one("h3.title-news a")
                if not a_tag:
                    continue
                title = a_tag.get_text(strip=True)
                link = a_tag["href"]

                if link in existing_links:
                    continue

                try:
                    detail_res = requests.get(link, headers=headers, timeout=10)
                    detail_soup = BeautifulSoup(detail_res.content, "html.parser")
                    content_tag = detail_soup.select_one(".fck_detail")
                    time_tag = detail_soup.select_one(".date")

                    if content_tag and time_tag:
                        content = content_tag.get_text(strip=True)
                        time = time_tag.get_text(strip=True)

                        if is_related_to_hanoi(title, content):
                            data.append({
                                "title": title,
                                "time": time,
                                "category": "VnExpress",
                                "content": content,
                                "link": link
                            })
                            existing_links.add(link)
                except Exception as e:
                    print(f"❌ Chi tiết VnExpress lỗi: {link} - {e}")
        except Exception as e:
            print(f"❌ Trang VnExpress lỗi: {url} - {e}")
            continue
    return data

# Crawl VnExpress Hà Nội (20 trang)
def crawl_vnexpress_ha_noi(max_pages=20, existing_links=None):
    if existing_links is None:
        existing_links = set()
    base_url = "https://vnexpress.net/topic/ha-noi-26482"
    headers = {"User-Agent": "Mozilla/5.0"}
    data = []

    for i in range(1, max_pages + 1):
        url = base_url if i == 1 else f"{base_url}-p{i}"
        print(f"🌐 VnExpress Hà Nội: {url}")
        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")
            articles = soup.select(".featured-item, .item-news")

            for article in articles:
                a_tag = article.select_one("h3.title-news a")
                if not a_tag:
                    continue
                title = a_tag.get_text(strip=True)
                link = a_tag["href"]

                if link in existing_links:
                    continue

                try:
                    detail_res = requests.get(link, headers=headers, timeout=10)
                    detail_soup = BeautifulSoup(detail_res.content, "html.parser")
                    content_tag = detail_soup.select_one(".fck_detail")
                    time_tag = detail_soup.select_one(".date")

                    if content_tag and time_tag:
                        content = content_tag.get_text(strip=True)
                        time = time_tag.get_text(strip=True)

                        if is_related_to_hanoi(title, content):
                            data.append({
                                "title": title,
                                "time": time,
                                "category": "VnExpress",
                                "content": content,
                                "link": link
                            })
                            existing_links.add(link)
                except Exception as e:
                    print(f"❌ Chi tiết VnExpress lỗi: {link} - {e}")
        except Exception as e:
            print(f"❌ Trang VnExpress lỗi: {url} - {e}")
            continue
    return data

# ==== Crawl Dân Trí (30 trang) ====
def crawl_dantri(max_pages=30, existing_links=None):
    if existing_links is None:
        existing_links = set()
    headers = {"User-Agent": "Mozilla/5.0"}
    data = []
    seen_links = set()

    for page in range(1, max_pages + 1):
        if page == 1:
            url = "https://dantri.com.vn/xa-hoi/moi-truong.htm"
        else:
            url = f"https://dantri.com.vn/xa-hoi/moi-truong/trang-{page}.htm"

        print(f"🌐 Dân Trí: {url}")
        try:
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.content, "html.parser")
            articles = soup.select("article")

            for article in articles:
                a_tag = article.select_one("h3 a")
                if not a_tag:
                    continue
                link = a_tag["href"]
                title = a_tag.get_text(strip=True)

                # 🛑 Loại trùng URL
                if link in seen_links or link in existing_links:
                    continue
                seen_links.add(link)
                existing_links.add(link)

                try:
                    detail_res = requests.get(link, headers=headers)
                    detail_soup = BeautifulSoup(detail_res.content, "html.parser")
                    content_tag = detail_soup.select_one(".singular-content")
                    time_tag = detail_soup.select_one(".author-time")

                    if content_tag and time_tag:
                        content = content_tag.get_text(strip=True)
                        time = time_tag.get_text(strip=True)

                        if is_related_to_hanoi(title, content):
                            data.append({
                                "title": title,
                                "time": time,
                                "category": "Dân Trí",
                                "content": content,
                                "link": link
                            })
                except Exception as e:
                    print(f"❌ Chi tiết Dân Trí lỗi: {link} - {e}")
        except Exception as e:
            print(f"❌ Trang Dân Trí lỗi: {url} - {e}")
    return data

# ==== Hàm chuẩn hóa thời gian ====
def standardize_time(time_str):
    """Chuẩn hóa định dạng thời gian từ các báo khác nhau về dạng YYYY-MM-DD"""
    if not time_str:
        return datetime.now().strftime("%Y-%m-%d")
        
    time_str = time_str.lower().strip()
    
    # Xử lý định dạng "Thứ ..., DD/MM/YYYY, HH:MM (GMT+7)"
    match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', time_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Xử lý định dạng "HH:MM - Ngày DD/MM/YYYY"
    match = re.search(r'ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})', time_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Xử lý định dạng chỉ có ngày và tháng của năm hiện tại (DD/MM)
    match = re.search(r'(\d{1,2})[/-](\d{1,2})', time_str)
    if match:
        # Trong tiếng Việt, định dạng thường là ngày/tháng
        day, month = int(match.group(1)), int(match.group(2))
        
        # Kiểm tra và đảm bảo day <= 31 và month <= 12
        if day > 31 or month > 12:
            # Nếu có lỗi, đảo ngược day và month
            day, month = month, day
            
        # Nếu vẫn không hợp lệ, sử dụng giá trị an toàn
        day = min(day, 31)
        month = min(month, 12)
        
        current_year = datetime.now().year
        return f"{current_year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    
    # Nếu không khớp với bất kỳ mẫu nào, trả về thời gian hiện tại
    return datetime.now().strftime("%Y-%m-%d")

# ==== Lưu dữ liệu vào file JSON ====
def save_to_json(data, filename):
    # Đọc dữ liệu cũ nếu file tồn tại
    existing_data = []
    existing_links = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            for item in existing_data:
                if "link" in item:
                    existing_links.add(item["link"])
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Kết hợp dữ liệu cũ và mới, loại trùng theo link
    new_data = []
    for item in data:
        if "link" in item and item["link"] not in existing_links:
            new_data.append(item)
            existing_links.add(item["link"])
    combined_data = existing_data + new_data
    
    # Lưu dữ liệu vào file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

# ==== Hàm lọc bài viết liên quan đến không khí ====
def filter_air_articles(input_file, output_file):
    filtered = []

    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            data = json.load(infile)
            
            for item in data:
                title = item.get("title", "")
                content = item.get("content", "")
                time_str = item.get("time", "")
                
                # Chuẩn hóa thời gian
                item["standardized_time"] = standardize_time(time_str)
                
                matched_keywords = is_air_related(title, content)

                if matched_keywords:
                    item["keyword"] = ", ".join(matched_keywords)
                    filtered.append(item)

        if filtered:
            with open(output_file, "w", encoding="utf-8") as outfile:
                json.dump(filtered, outfile, ensure_ascii=False, indent=2)
            print(f"✅ Đã lọc {len(filtered)} bài liên quan đến không khí vào '{output_file}'")
            print(f"✅ Đã chuẩn hóa cột thời gian cho tất cả các bài viết")
        else:
            print("⚠️ Không tìm thấy bài nào liên quan.")
    except Exception as e:
        print(f"❌ Lỗi khi lọc bài viết: {e}")

# ==== Hàm lưu dữ liệu vào database ====
def save_to_database(json_file):
    """Lưu dữ liệu từ file JSON vào TimescaleDB"""
    try:
        # Đọc dữ liệu từ file JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("⚠️ Không có dữ liệu để lưu vào database")
            return False
        
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame(data)
        
        # Chuyển đổi cột thời gian
        df['time'] = pd.to_datetime(df['standardized_time'])
        
        # Tạo kết nối đến TimescaleDB
        db_util = TimescaleDBUtil()
        if not db_util.connect():
            print("❌ Không thể kết nối đến TimescaleDB")
            return False
        
        # Lọc các cột cần thiết
        if 'standardized_time' in df.columns:
            # Chuẩn bị dữ liệu để lưu vào database
            news_df = df[['title', 'time', 'category', 'content', 'link', 'keyword']]
        else:
            print("❌ Dữ liệu không có cột standardized_time")
            return False
        
        # Lưu vào database
        result = db_util.create_table_from_dataframe(
            df=news_df,
            table_name="news",
            time_column="time",
            schema="public",
            if_exists="append"
        )
        
        # Đóng kết nối
        db_util.disconnect()
        
        if result:
            print(f"✅ Đã lưu {len(news_df)} bài viết vào bảng news trong database")
            return True
        else:
            print("❌ Lưu dữ liệu vào database thất bại")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi lưu dữ liệu vào database: {e}")
        return False

# ==== Hàm chính ====
def main():
    # Đường dẫn file dữ liệu
    raw_data_file = "data/air_quality_news.json"
    filtered_data_file = "data/air_quality_filtered.json"
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("data", exist_ok=True)

    # Bước 1: Crawl dữ liệu
    print("🗂 Đang đọc dữ liệu đã lưu...")
    existing_links = read_existing_links(raw_data_file)

    print("🚀 Bắt đầu crawl các báo...")
    all_data = []
    today = datetime.now().strftime("%Y-%m-%d")
    
    for func in [crawl_vnexpress, crawl_vnexpress_ha_noi, crawl_dantri]:
        try:
            raw_data = func(existing_links=existing_links)
            for item in raw_data:
                item["crawled_date"] = today
                all_data.append(item)
                if "link" in item:
                    existing_links.add(item["link"])
        except Exception as e:
            print(f"⚠️ Lỗi khi chạy hàm {func.__name__}: {e}")

    # Lưu dữ liệu vào file JSON
    save_to_json(all_data, raw_data_file)
    print(f"✅ Đã lưu {len(all_data)} bài mới vào {raw_data_file}")
    
    # Bước 2: Lọc bài viết liên quan đến không khí
    print("🔍 Đang lọc bài viết liên quan đến không khí...")
    filter_air_articles(raw_data_file, filtered_data_file)
    
    # Bước 3: Lưu vào database
    print("💾 Đang lưu dữ liệu vào database...")
    save_to_database(filtered_data_file)

if __name__ == "__main__":
    main() 