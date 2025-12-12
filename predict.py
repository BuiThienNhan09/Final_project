import argparse
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 1. Cấu hình đường dẫn
MODEL_PATH = 'flight_price_model.keras'
PREPROCESSOR_PATH = 'datasets/processed/preprocessor.pkl'
METADATA_PATH = 'datasets/processed/preprocessor_metadata.json'

# --- Load model & preprocessor once at import-time ---
model = None
preprocessor = None
metadata = {}
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    # Defer hard failure until predict is called; keep helpful message
    model = None

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception:
    preprocessor = None

if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}


def predict_fare(airline, source_city, departure_time, stops, arrival_time, destination_city, class_type, duration, days_left):
    """
    Hàm nhập thông tin chuyến bay và trả về giá dự đoán.
    """
    
    # --- A. Tải Model và Preprocessor ---
    # Chỉ tải 1 lần nếu chạy web app, nhưng ở script đơn giản ta tải luôn tại đây
    # Ensure model & preprocessor are loaded
    global model, preprocessor
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Lỗi: Không tải được model từ {MODEL_PATH}. Hãy chạy train.py trước!\n{e}")
            return
    if preprocessor is None:
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        except Exception as e:
            print(f"Lỗi: Không tải được preprocessor từ {PREPROCESSOR_PATH}. Hãy chạy preprocess.py trước!\n{e}")
            return

    # --- B. Tạo DataFrame từ dữ liệu nhập vào ---
    # Tên cột phải GIỐNG HỆT lúc train (trong file Clean_Dataset.csv)
    input_data = pd.DataFrame({
        'airline': [airline],
        'source_city': [source_city],
        'departure_time': [departure_time],
        'stops': [stops],
        'arrival_time': [arrival_time],
        'destination_city': [destination_city],
        'class': [class_type],
        'duration': [duration],
        'days_left': [days_left]
    })

    print("-" * 30)
    print("Thông tin vé cần dự đoán:")
    print(input_data.iloc[0].to_string())

    # --- C. Tiền xử lý dữ liệu mới ---
    # Dùng preprocessor đã lưu để biến đổi dữ liệu mới y hệt cách làm với tập train
    try:
        input_processed = preprocessor.transform(input_data)
    except Exception as e:
        # Try to provide helpful info from metadata
        msg = f"\nLỗi xử lý dữ liệu: Có thể bạn nhập giá trị không tồn tại trong tập train (ví dụ tên hãng bay lạ).\n{e}\n"
        if metadata:
            cat_features = metadata.get('cat_features')
            encoder_cats = metadata.get('encoder_categories')
            if cat_features and encoder_cats:
                msg += "\nCác giá trị hợp lệ theo cột phân loại:\n"
                for col, cats in zip(cat_features, encoder_cats):
                    msg += f" - {col}: {cats[:10]}{'...' if len(cats)>10 else ''}\n"
        print(msg)
        return

    # --- D. Dự đoán ---
    prediction = model.predict(input_processed)
    # Model was trained on log1p(target) -> inverse with expm1
    try:
        predicted_price = np.expm1(prediction[0][0])
    except Exception:
        predicted_price = float(prediction[0][0])

    print("-" * 30)
    print(f"💰 GIÁ VÉ DỰ ĐOÁN: {predicted_price:,.2f} (Đơn vị tiền tệ gốc)")
    print("-" * 30)

    return predicted_price

# --- CHẠY THỬ NGHIỆM ---
def _predict_df(df: pd.DataFrame):
    """Predict for a DataFrame (returns a numpy array of predicted prices on original scale)."""
    # Validate columns
    required_cols = ['airline','source_city','departure_time','stops','arrival_time','destination_city','class','duration','days_left']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    global preprocessor, model
    if preprocessor is None:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    X_proc = preprocessor.transform(df[required_cols])
    preds_log = model.predict(X_proc)
    try:
        preds = np.expm1(preds_log.flatten())
    except Exception:
        preds = preds_log.flatten().astype(float)
    return preds


def _cli():
    parser = argparse.ArgumentParser(description='Predict flight price (single row or CSV input)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--input-file', '-i', help='CSV file with rows to predict')
    group.add_argument('--single', action='store_true', help='Use CLI flags to provide a single sample')

    # Single-row args
    parser.add_argument('--airline', default='Vistara')
    parser.add_argument('--source_city', default='Delhi')
    parser.add_argument('--departure_time', default='Morning')
    parser.add_argument('--stops', default='one')
    parser.add_argument('--arrival_time', default='Night')
    parser.add_argument('--destination_city', default='Mumbai')
    parser.add_argument('--class_type', dest='class_type', default='Business')
    parser.add_argument('--duration', type=float, default=14.5)
    parser.add_argument('--days_left', type=int, default=20)
    parser.add_argument('--output-file', '-o', help='If input-file provided, save predictions to this CSV path')

    args = parser.parse_args()
    if args.input_file:
        df = pd.read_csv(args.input_file)
        preds = _predict_df(df)
        df['predicted_price'] = preds
        out = args.output_file or (os.path.splitext(args.input_file)[0] + '_predicted.csv')
        df.to_csv(out, index=False)
        print(f"Saved predictions to {out}")
    else:
        # Build single-row DataFrame
        if args.single:
            row = {
                'airline': args.airline,
                'source_city': args.source_city,
                'departure_time': args.departure_time,
                'stops': args.stops,
                'arrival_time': args.arrival_time,
                'destination_city': args.destination_city,
                'class': args.class_type,
                'duration': args.duration,
                'days_left': args.days_left
            }
        else:
            # Default demo row (backwards compatible)
            row = {
                'airline': 'Vistara',
                'source_city': 'Delhi',
                'departure_time': 'Morning',
                'stops': 'one',
                'arrival_time': 'Night',
                'destination_city': 'Mumbai',
                'class': 'Business',
                'duration': 14.5,
                'days_left': 20
            }
        df = pd.DataFrame([row])
        pred = _predict_df(df)[0]
        print(f"Predicted price: {pred:,.2f}")


if __name__ == '__main__':
    _cli()