from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- CẤU HÌNH ---
MODEL_PATH = 'flight_price_model.keras'
PREPROCESSOR_PATH = 'datasets/processed/preprocessor.pkl'

# --- LOAD MODEL ---
print("⏳ Đang tải hệ thống AI...")
try:
    model = load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("✅ Đã tải xong Model và Preprocessor!")
except Exception as e:
    print(f"❌ Lỗi tải file: {e}")
    model = None
    preprocessor = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not preprocessor:
        return render_template('index.html', prediction_text="Lỗi: Không tìm thấy Model.")

    try:
        # 1. Lấy dữ liệu
        input_data = pd.DataFrame({
            'airline': [request.form['airline']],
            'source_city': [request.form['source_city']],
            'departure_time': [request.form['departure_time']],
            'stops': [request.form['stops']],
            'arrival_time': [request.form['arrival_time']],
            'destination_city': [request.form['destination_city']],
            'class': [request.form['flight_class']],
            'duration': [float(request.form['duration'])],
            'days_left': [int(request.form['days_left'])]
        })

        # 2. Xử lý & Dự đoán
        processed_data = preprocessor.transform(input_data)
        pred_log = model.predict(processed_data)
        
        # --- 3. QUY ĐỔI TIỀN TỆ (PHẦN MỚI THÊM) ---
        price_inr = np.expm1(pred_log[0][0])       # Giá gốc Rupee
        price_vnd = price_inr * 300                # 1 INR ≈ 300 VND
        price_usd = price_inr * 0.0118             # 1 INR ≈ 0.0118 USD

        # --- 4. FORMAT HIỂN THỊ ---
        # Hiển thị dòng chính là VND cho dễ nhìn
        msg_main = f"{price_vnd:,.0f} VND"
        
        # Hiển thị dòng phụ là USD và INR
        msg_sub = f"(${price_usd:,.2f} USD | {price_inr:,.0f} INR)"

        # Logic lời khuyên
        advice = "✅ Giá tiêu chuẩn"
        advice_class = "normal"
        
        if price_inr > 40000:
            advice = "💎 Vé Thương Gia (Giá Cao)"
            advice_class = "expensive"
        elif price_inr < 5000:
            advice = "🔥 Vé Siêu Rẻ (Nên Mua)"
            advice_class = "cheap"

        # Truyền cả 2 dòng giá sang HTML
        return render_template('index.html', 
                               prediction_text=msg_main, 
                               sub_text=msg_sub,   # <--- Biến mới chứa USD/INR
                               advice_text=advice,
                               advice_class=advice_class,
                               show_result=True)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Lỗi: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)