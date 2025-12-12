import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import joblib

# 1. Cấu hình
DATA_PATH = 'datasets/Clean_Dataset.csv'
PROCESSED_FOLDER = 'datasets/processed'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Tạo thư mục lưu dữ liệu đã xử lý nếu chưa có
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def load_and_preprocess():
    print("Example: Đang tải dữ liệu...")
    df = pd.read_csv(DATA_PATH)
    
    # --- A. Xử lý sơ bộ ---
    # Bỏ cột 'Unnamed: 0' nếu có (thường sinh ra khi lưu csv từ pandas)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Bỏ cột 'flight' vì mã chuyến bay thường không ảnh hưởng giá chung (hoặc quá nhiều unique value)
    if 'flight' in df.columns:
        df = df.drop('flight', axis=1)

    print(f"Dữ liệu gốc: {df.shape}")

    # --- B. Tách Feature (X) và Target (y) ---
    y = df['price'] # Target dự đoán
    X = df.drop('price', axis=1) # Các đặc trưng đầu vào

    # --- C. Phân loại cột để xử lý ---
    # Cột phân loại (chữ) -> Cần chuyển thành số (One-Hot Encoding)
    # Lưu ý: Cột 'class' (Economy/Business) có thể map thủ công nếu muốn thứ tự, 
    # nhưng ở đây ta dùng OneHot cho tiện.
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    
    # Cột số -> Cần chuẩn hóa (Scaling) về khoảng 0-1 để ANN học tốt hơn
    numerical_cols = [col for col in X.columns if X[col].dtype != 'object']

    print(f"Cột số: {numerical_cols}")
    print(f"Cột phân loại: {categorical_cols}")

    # --- D. Pipeline xử lý ---
    # Sử dụng ColumnTransformer để xử lý song song
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

    # Fit và transform dữ liệu X
    print("Đang xử lý dữ liệu (Encoding & Scaling)...")
    X_processed = preprocessor.fit_transform(X)
    
    # Lưu lại scaler/encoder để dùng sau này khi dự đoán thực tế
    joblib.dump(preprocessor, f'{PROCESSED_FOLDER}/preprocessor.pkl')

    # --- E. Chia tập Train / Test ---
    print("Đang chia tập Train/Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --- F. Lưu dữ liệu đã xử lý ---
    # Lưu dưới dạng numpy array (.npy) để load nhanh hơn khi train
    np.save(f'{PROCESSED_FOLDER}/X_train.npy', X_train)
    np.save(f'{PROCESSED_FOLDER}/X_test.npy', X_test)
    np.save(f'{PROCESSED_FOLDER}/y_train.npy', y_train)
    np.save(f'{PROCESSED_FOLDER}/y_test.npy', y_test)

    print(f"Hoàn tất! Dữ liệu đã lưu tại: {PROCESSED_FOLDER}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

if __name__ == "__main__":
    load_and_preprocess()