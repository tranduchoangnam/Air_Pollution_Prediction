

1. Clone repository về máy:

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Cài đặt Chromium cho Playwright:
```bash
playwright install chromium
```

### Chạy dự án

1. Chạy script crawl dữ liệu:
```bash
python crawl_iqair.py
```

2. Dữ liệu sau khi crawl sẽ được lưu vào thư mục `result/` dưới dạng file CSV
