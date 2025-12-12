import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# 1. Cấu hình
PROCESSED_FOLDER = 'datasets/processed'
MODEL_PATH = 'flight_price_model.keras' # Đuôi .keras là chuẩn mới của TensorFlow
BATCH_SIZE = 32
# Allow overriding epochs via environment variable for quick tests
EPOCHS = int(os.getenv('EPOCHS', '100')) # Số lần học lặp lại tối đa

def train_model():
    # --- A. Tải dữ liệu đã xử lý ---
    print("Đang tải dữ liệu huấn luyện...")
    X_train = np.load(f'{PROCESSED_FOLDER}/X_train.npy')
    y_train = np.load(f'{PROCESSED_FOLDER}/y_train.npy')
    X_test = np.load(f'{PROCESSED_FOLDER}/X_test.npy')
    y_test = np.load(f'{PROCESSED_FOLDER}/y_test.npy')

    # --- Tạo tập validation từ X_train để có validation_data rõ ràng ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # --- Lưu giá trị y gốc để tính metric trên scale thực ---
    y_train_orig = y_train.copy()
    y_val_orig = y_val.copy()
    y_test_orig = y_test.copy()

    # --- Tiền xử lý target: log-transform để ổn định phương sai và giảm ảnh hưởng outlier ---
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)
    y_test = np.log1p(y_test)

    input_dim = X_train.shape[1] # Số lượng đặc trưng đầu vào (số cột)
    print(f"Số lượng đặc trưng đầu vào (Input features): {input_dim}")

    # --- B. Xây dựng cấu trúc mạng ANN ---
    # Cấu trúc: Input -> Lớp ẩn 1 -> Lớp ẩn 2 -> Lớp ẩn 3 -> Output
    model = Sequential()

    # Lớp ẩn 1: 128 nơ-ron + BatchNorm + Dropout
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Lớp ẩn 2: 64 nơ-ron
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Lớp ẩn 3: 32 nơ-ron
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())

    # Lớp Output: 1 nơ-ron (vì dự đoán 1 giá tiền duy nhất)
    # Không dùng hàm kích hoạt (activation=None) hoặc dùng 'linear' cho bài toán hồi quy
    model.add(Dense(1, activation='linear'))

    # Thiết lập cách học (Optimizer & Loss function)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    model.summary() # In cấu trúc mạng ra màn hình

    # --- C. Huấn luyện (Training) ---
    print("\nBắt đầu huấn luyện...")
    
    # Tự động dừng nếu mô hình không cải thiện sau 10 lần lặp (Early Stopping)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Lưu lại model tốt nhất trong quá trình train
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    
    # Giảm learning rate khi val_loss không cải thiện
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # --- Callback tính metric trên scale gốc mỗi epoch ---
    class MetricsOnOriginalScale(tf.keras.callbacks.Callback):
        def __init__(self, X_val, y_val_orig, patience=5, r2_threshold=None):
            """Callback tính metric trên scale gốc mỗi epoch.
            - patience: số epoch không cải thiện R2 trước khi dừng.
            - r2_threshold: nếu R2 >= threshold sẽ dừng ngay.
            """
            super().__init__()
            self.X_val = X_val
            self.y_val_orig = np.array(y_val_orig).flatten()
            self.val_r2 = []
            self.val_mae = []
            self.val_rmse = []
            self.patience = patience
            self.r2_threshold = r2_threshold
            self.best_r2 = -np.inf
            self.wait = 0

        def on_epoch_end(self, epoch, logs=None):
            # Dự đoán trên validation (model đang huấn luyện với target đã log)
            y_pred_log = self.model.predict(self.X_val, verbose=0)
            # Đảo ngược log-transform
            try:
                y_pred = np.expm1(y_pred_log).flatten()
            except Exception:
                y_pred = y_pred_log.flatten()

            y_true = self.y_val_orig
            # Tính metrics trên scale gốc
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            self.val_r2.append(r2)
            self.val_mae.append(mae)
            self.val_rmse.append(rmse)
            # In tóm tắt mỗi epoch để tiện theo dõi
            print(f"[Epoch {epoch+1}] val_r2: {r2:.4f}, val_mae: {mae:.2f}, val_rmse: {rmse:.2f}")

            # Early-stop logic based on R2
            # Nếu đạt threshold thì dừng ngay
            if (self.r2_threshold is not None) and (r2 >= self.r2_threshold):
                print(f"Stopping training: val_r2 {r2:.4f} >= threshold {self.r2_threshold}")
                self.model.stop_training = True
                return

            # Nếu cải thiện R2 thì reset wait
            if r2 > self.best_r2 + 1e-6:
                self.best_r2 = r2
                self.wait = 0
            else:
                self.wait += 1

            # Nếu vượt patience, dừng
            if self.wait >= self.patience:
                print(f"Stopping training: no improvement in val_r2 for {self.patience} epochs (best={self.best_r2:.4f})")
                self.model.stop_training = True
                return

    metrics_on_orig = MetricsOnOriginalScale(X_val, y_val_orig)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, reduce_lr, metrics_on_orig],
        verbose=1
    )

    # --- D. Đánh giá (Evaluation) ---
    print("\nĐang đánh giá trên tập Test...")
    y_pred = model.predict(X_test)
    # Đảo ngược dự đoán về scale gốc trước khi tính metric cuối cùng
    try:
        y_pred = np.expm1(y_pred).flatten()
        y_true = np.array(y_test_orig).flatten()
    except Exception:
        y_pred = y_pred.flatten()
        y_true = np.array(y_test).flatten()

    # Tính các chỉ số đánh giá trên scale gốc
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("-" * 30)
    print(f"KẾT QUẢ ĐÁNH GIÁ:")
    print(f"R2 Score (Độ chính xác): {r2:.4f} (Càng gần 1 càng tốt)")
    print(f"MAE (Sai số tuyệt đối trung bình): {mae:.2f}")
    print(f"RMSE (Căn bậc hai sai số bình phương TB): {rmse:.2f}")
    print("-" * 30)

    # --- E. Vẽ biểu đồ Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Quá trình huấn luyện (Model Loss)')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('datasets/training_history.png')
    print("Đã lưu biểu đồ huấn luyện tại: datasets/training_history.png")

    # --- Vẽ thêm metric trên scale gốc lưu bởi callback ---
    try:
        epochs = range(1, len(metrics_on_orig.val_mae) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics_on_orig.val_mae, label='Val MAE (original scale)')
        plt.plot(epochs, metrics_on_orig.val_rmse, label='Val RMSE (original scale)')
        plt.title('Validation metrics on original scale per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('datasets/validation_metrics_original.png')
        print('Đã lưu biểu đồ metric trên scale gốc tại: datasets/validation_metrics_original.png')
    except Exception:
        pass

    # --- Vẽ R2 trên scale gốc mỗi epoch (nếu có) ---
    try:
        epochs = range(1, len(metrics_on_orig.val_r2) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics_on_orig.val_r2, marker='o', label='Val R2 (original scale)')
        plt.title('Validation R2 on original scale per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('R2')
        plt.ylim(-0.1, 1.0)
        plt.legend()
        plt.savefig('datasets/validation_r2_per_epoch.png')
        print('Đã lưu biểu đồ R2 trên scale gốc tại: datasets/validation_r2_per_epoch.png')
    except Exception:
        pass

if __name__ == "__main__":
    train_model()