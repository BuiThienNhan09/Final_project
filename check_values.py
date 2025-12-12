import pandas as pd

# Load dữ liệu gốc
df = pd.read_csv('datasets/Clean_Dataset.csv')

print("--- DANH SÁCH CÁC GIÁ TRỊ HỢP LỆ ---")
cols_to_check = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']

for col in cols_to_check:
    if col in df.columns:
        print(f"\nCột [{col}]:")
        # In ra các giá trị duy nhất (unique)
        print(df[col].unique())
    else:
        print(f"\nCảnh báo: Không tìm thấy cột {col}")