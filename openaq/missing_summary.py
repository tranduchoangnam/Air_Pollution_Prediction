import pandas as pd

# Đọc file CSV đã xử lý
input_file = 'processed_output.csv'
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error reading {input_file}: {e}")
    exit()

# Tính số lượng giá trị bị thiếu cho từng cột
missing_data = df.isna().sum()

# Tạo DataFrame cho báo cáo
missing_report = pd.DataFrame({
    'Column': missing_data.index,
    'Missing_Values': missing_data.values
})

# In báo cáo ra màn hình
print("\nMissing Data Summary:")
print(missing_report.to_string(index=False))

# Lưu báo cáo vào file CSV
output_file = 'missing_data_report.csv'
missing_report.to_csv(output_file, index=False, encoding='utf-8')
print(f"\nMissing data report saved as {output_file}")