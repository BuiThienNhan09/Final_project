import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # <-- Thêm thư viện 'os' để xử lý thư mục

# --- Cài đặt chung cho biểu đồ ---
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 14

# --- Thiết lập thư mục đầu ra ---
output_folder = 'EDA_plots'
# Tạo thư mục nếu nó chưa tồn tại
os.makedirs(output_folder, exist_ok=True) 
print(f"Tất cả biểu đồ sẽ được lưu vào thư mục: '{output_folder}/'")

# --- Tải dữ liệu ---
try:
    df = pd.read_csv('datasets/Clean_Dataset.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp 'datasets/Clean_Dataset.csv'.")
    exit()

# --- Bước làm sạch nhỏ ---
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# --- Biểu đồ 1: Phân phối của Giá vé (Thường và Log) ---
print("Đang tạo Biểu đồ 1: Phân phối giá vé...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.histplot(df['price'], kde=True, bins=50, ax=axes[0])
axes[0].set_title('Phân phối của Giá vé (Price Distribution)')
axes[0].set_xlabel('Giá vé')
axes[0].set_ylabel('Tần suất')

sns.histplot(np.log1p(df['price']), kde=True, bins=50, ax=axes[1])
axes[1].set_title('Phân phối của Log(Giá vé) (Log-Transformed Price)')
axes[1].set_xlabel('Log(Giá vé)')
axes[1].set_ylabel('Tần suất')

plt.tight_layout()
# Cập nhật đường dẫn lưu
save_path_1 = os.path.join(output_folder, 'price_distribution_charts.png')
plt.savefig(save_path_1)
plt.close(fig)

# --- Biểu đồ 2: Giá vé theo Hạng vé (Price by Class) ---
print("Đang tạo Biểu đồ 2: Giá vé theo Hạng vé...")
plt.figure(figsize=(10, 7))
sns.boxplot(x='class', y='price', data=df, palette=['#a8dda8', '#e2a3c7'])
plt.title('Giá vé theo Hạng vé (Price by Class)', fontsize=16)
plt.xlabel('Hạng vé', fontsize=12)
plt.ylabel('Giá vé', fontsize=12)
# Cập nhật đường dẫn lưu
save_path_2 = os.path.join(output_folder, 'price_vs_class.png')
plt.savefig(save_path_2)
plt.close()

# --- Biểu đồ 3: Giá vé theo Hãng hàng không (tách theo Hạng vé) ---
print("Đang tạo Biểu đồ 3: Giá vé theo Hãng hàng không...")
plt.figure(figsize=(16, 9))
sns.boxplot(x='airline', y='price', hue='class', data=df, palette=['#a8dda8', '#e2a3c7'])
plt.title('Giá vé theo Hãng hàng không (tách theo Hạng vé)', fontsize=16)
plt.xlabel('Hãng hàng không', fontsize=12)
plt.ylabel('Giá vé', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Hạng vé')
# Cập nhật đường dẫn lưu
save_path_3 = os.path.join(output_folder, 'price_vs_airline_by_class.png')
plt.savefig(save_path_3)
plt.close()

# --- Biểu đồ 4: Giá vé theo Số điểm dừng (tách theo Hạng vé) ---
print("Đang tạo Biểu đồ 4: Giá vé theo Số điểm dừng...")
plt.figure(figsize=(12, 8))
stop_order = ['zero', 'one', 'two_or_more']
sns.boxplot(x='stops', y='price', hue='class', data=df, order=stop_order, palette=['#a8dda8', '#e2a3c7'])
plt.title('Giá vé theo Số điểm dừng (tách theo Hạng vé)', fontsize=16)
plt.xlabel('Số điểm dừng', fontsize=12)
plt.ylabel('Giá vé', fontsize=12)
plt.legend(title='Hạng vé')
# Cập nhật đường dẫn lưu
save_path_4 = os.path.join(output_folder, 'price_vs_stops_by_class.png')
plt.savefig(save_path_4)
plt.close()

# --- Biểu đồ 5: Thời gian bay (Duration) vs. Giá vé (Price) ---
print("Đang tạo Biểu đồ 5: Thời gian bay vs. Giá vé...")
plt.figure(figsize=(16, 9))
sns.scatterplot(x='duration', y='price', hue='class', data=df, alpha=0.5, palette=['#a8dda8', '#e2a3c7'])
plt.title('Thời gian bay (Duration) vs. Giá vé (Price)', fontsize=16)
plt.xlabel('Thời gian bay (giờ)', fontsize=12)
plt.ylabel('Giá vé', fontsize=12)
plt.legend(title='Hạng vé')
# Cập nhật đường dẫn lưu
save_path_5 = os.path.join(output_folder, 'price_vs_duration_by_class.png')
plt.savefig(save_path_5)
plt.close()

# --- Biểu đồ 6: Giá vé trung bình theo Số ngày còn lại (Days Left) ---
print("Đang tạo Biểu đồ 6: Giá vé theo Số ngày còn lại...")
df_days_mean = df.groupby(['days_left', 'class'])['price'].mean().reset_index()

plt.figure(figsize=(16, 9))
sns.lineplot(x='days_left', y='price', hue='class', data=df_days_mean, marker='o', palette=['#a8dda8', '#e2a3c7'])
plt.title('Giá vé trung bình theo Số ngày còn lại (tách theo Hạng vé)', fontsize=16)
plt.xlabel('Số ngày còn lại trước chuyến bay', fontsize=12)
plt.ylabel('Giá vé trung bình', fontsize=12)
plt.legend(title='Hạng vé')
plt.gca().invert_xaxis()
# Cập nhật đường dẫn lưu
save_path_6 = os.path.join(output_folder, 'price_vs_days_left_by_class.png')
plt.savefig(save_path_6)
plt.close()

print("\n--- HOÀN THÀNH ---")
print(f"Đã tạo và lưu thành công 6 biểu đồ vào thư mục '{output_folder}/'.")