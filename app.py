from flask import Flask, render_template, request, jsonify
import os
import sys

# Reduce TensorFlow memory usage BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow with memory optimization
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Limit TensorFlow memory growth
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# Import database module
from database import db as search_db

app = Flask(__name__)

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📁 Base directory: {BASE_DIR}")

# ============================================================
# LOCATION MAPPING (CHUẨN HÓA ĐỊA ĐIỂM)
# ============================================================
LOCATION_MAPPING = {
    # Miền Nam
    'TP HCM': 'SGN', 'Hồ Chí Minh': 'SGN', 'Hồ Chí Minh (SGN)': 'SGN', 'SGN': 'SGN',
    'Phú Quốc': 'PQC', 'Phú Quốc (PQC)': 'PQC', 'PQC': 'PQC', 'PHÚ QUỐC': 'PQC',
    'Cần Thơ': 'VCA', 'CẦN THƠ': 'VCA', 'VCA': 'VCA',
    'Côn Đảo': 'VCS', 'VCS': 'VCS',
    'Rạch Giá': 'VKG', 'VKG': 'VKG',

    # Miền Trung
    'Đà Nẵng': 'DAD', 'ĐÀ NẴNG': 'DAD', 'Đà Nẵng (DAD)': 'DAD', 'DAD': 'DAD',
    'Nha Trang': 'CXR', 'NHA TRANG': 'CXR', 'Nha Trang (CXR)': 'CXR', 'Cam Ranh': 'CXR', 'CXR': 'CXR',
    'Đà Lạt': 'DLI', 'ĐÀ LẠT': 'DLI', 'Đà Lạt (DLI)': 'DLI', 'DLI': 'DLI',
    'Huế': 'HUI', 'HUẾ': 'HUI', 'Huế (HUI)': 'HUI', 'HUI': 'HUI',
    'Quy Nhơn': 'UIH', 'QUY NHƠN': 'UIH', 'UIH': 'UIH',
    'Vinh': 'VII', 'VINH': 'VII', 'Vinh (VII)': 'VII', 'VII': 'VII',
    'Thanh Hóa': 'THD', 'THANH HÓA': 'THD', 'THD': 'THD',
    'Chu Lai': 'VCL', 'CHU LAI': 'VCL', 'VCL': 'VCL', 'Quảng Nam': 'VCL',
    'Tuy Hòa': 'TBB', 'TBB': 'TBB',
    'Đồng Hới': 'VDH', 'VDH': 'VDH',
    'Buôn Ma Thuột': 'BMV', 'BMV': 'BMV',
    'Pleiku': 'PXU', 'PXU': 'PXU',

    # Miền Bắc
    'Hà Nội': 'HAN', 'HÀ NỘI': 'HAN', 'Hà Nội (HAN)': 'HAN', 'HAN': 'HAN',
    'Hải Phòng': 'HPH', 'HẢI PHÒNG': 'HPH', 'HPH': 'HPH',
    'Vân Đồn': 'VDO', 'VDO': 'VDO',
    'Điện Biên': 'DIN', 'DIN': 'DIN',

    # Quốc tế
    'Bangkok': 'BKK', 'BKK': 'BKK',
    'Singapore': 'SIN', 'SIN': 'SIN',
    'Kuala Lumpur': 'KUL', 'KUL': 'KUL',
    'Seoul': 'ICN', 'ICN': 'ICN',
    'Tokyo': 'NRT', 'NRT': 'NRT',
    'Taipei': 'TPE', 'TPE': 'TPE'
}

# ============================================================
# CLASS CATEGORY MAPPING
# ============================================================

CLASS_MAPPING = {
    'Economy': 'ECONOMY',
    'Business': 'BUSINESS'
}

def map_category_to_class(category):
    return CLASS_MAPPING.get(category, 'ECONOMY')

def get_category_from_class(class_name):
    class_upper = str(class_name).upper()
    if any(x in class_upper for x in ['BUSINESS', 'BUZ', 'SKYBOSS']):
        return 'Business'
    else:
        return 'Economy'

# ============================================================
# LOAD MODELS & ARTIFACTS
# ============================================================

print("\n" + "="*70)
print("🚀 KHỞI ĐỘNG FLIGHT PRICE PREDICTION SERVER (MULTI-MODEL)")
print("="*70)

# Dictionary to store all models
models = {}
scalers = {}
model_evaluation_results = {}

# Helper function for file paths
def get_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)

# Helper function to load JSON (try with and without .json extension)
def load_json_file(path):
    """Try to load JSON file with or without .json extension"""
    # Try with .json first
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Try without .json
    path_no_ext = path.replace('.json', '')
    if os.path.exists(path_no_ext):
        with open(path_no_ext, 'r', encoding='utf-8') as f:
            return json.load(f)
    raise FileNotFoundError(f"File not found: {path} or {path_no_ext}")

# Load ANN Model
print("\n📦 [1/3] Đang load ANN Model...")
try:
    models['ann'] = tf.keras.models.load_model(get_path('models/flight_price_model.h5'), compile=False)
    models['ann'].compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("   ✅ ANN Model loaded")
except Exception as e:
    print(f"   ❌ Lỗi load ANN model: {e}")
    models['ann'] = None

# Load ANN Scaler
try:
    with open(get_path('models/scaler.pkl'), 'rb') as f:
        scalers['ann'] = pickle.load(f)
    print("   ✅ ANN Scaler loaded")
except Exception as e:
    print(f"   ❌ Lỗi load ANN scaler: {e}")
    scalers['ann'] = None

# Load ANN Evaluation Results
try:
    ann_eval = load_json_file(get_path('models/evaluation_results.json'))
    if 'test' in ann_eval:
        model_evaluation_results['ann'] = ann_eval['test']
    else:
        model_evaluation_results['ann'] = ann_eval
    print(f"   ✅ ANN Accuracy: {model_evaluation_results['ann'].get('accuracy', 0):.2f}%")
except Exception as e:
    print(f"   ⚠️ ANN evaluation results not found: {e}")
    model_evaluation_results['ann'] = {}

# Load Linear Regression (Ridge) Model
print("\n📦 [2/3] Đang load Linear Regression Model...")
try:
    with open(get_path('models/ridge_regression_model.pkl'), 'rb') as f:
        models['linear_regression'] = pickle.load(f)
    print("   ✅ Linear Regression Model loaded")
    
    with open(get_path('models/ridge_scaler.pkl'), 'rb') as f:
        scalers['linear_regression'] = pickle.load(f)
    print("   ✅ Linear Regression Scaler loaded")
    
    lr_eval = load_json_file(get_path('models/ridge_evaluation_results.json'))
    model_evaluation_results['linear_regression'] = lr_eval
    print(f"   ✅ Linear Regression Accuracy: {lr_eval.get('accuracy', lr_eval.get('accuracy_20', 0)):.2f}%")
except Exception as e:
    print(f"   ⚠️ Linear Regression not available: {e}")
    models['linear_regression'] = None
    scalers['linear_regression'] = None
    model_evaluation_results['linear_regression'] = {}

# Load Decision Tree Model
print("\n📦 [3/3] Đang load Decision Tree Model...")
try:
    with open(get_path('models/decision_tree_fixed_overfitting.pkl'), 'rb') as f:
        models['decision_tree'] = pickle.load(f)
    print("   ✅ Decision Tree Model loaded")
    
    dt_eval = load_json_file(get_path('models/dt_fixed_overfitting_results.json'))
    model_evaluation_results['decision_tree'] = dt_eval
    print(f"   ✅ Decision Tree Accuracy: {dt_eval.get('accuracy_20', dt_eval.get('accuracy', 0)):.2f}%")
except Exception as e:
    print(f"   ⚠️ Decision Tree not available: {e}")
    models['decision_tree'] = None
    model_evaluation_results['decision_tree'] = {}

# Load shared label encoders
try:
    with open(get_path('models/label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
    print("\n✅ Label encoders loaded")
except Exception as e:
    print(f"\n❌ Error loading label encoders: {e}")
    label_encoders = {}

# Load feature names
try:
    feature_names = load_json_file(get_path('models/feature_names.json'))
    print("✅ Feature names loaded")
except Exception as e:
    print(f"❌ Error loading feature names: {e}")
    feature_names = []

# Load unique values
try:
    unique_values = load_json_file(get_path('models/unique_values.json'))
    print("✅ Unique values loaded")
except Exception as e:
    print(f"❌ Error loading unique values: {e}")
    unique_values = {}

# Load airport names
try:
    airport_names = load_json_file(get_path('airport_names.json'))
    print("✅ Airport names loaded")
except:
    airport_names = {}
    print("⚠️ Airport names fallback")

# Load original data for statistics
try:
    df_original = pd.read_csv(get_path('Flight_Price_Data_Enhanced_Up.csv'))
    print(f"✅ Data loaded: {len(df_original):,} rows")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    df_original = pd.DataFrame()

# Get default accuracy (ANN)
accuracy = model_evaluation_results.get('ann', {}).get('accuracy', 0)

# Build valid routes dictionary
print("\n📋 Đang tạo danh sách routes hợp lệ...")
valid_routes = {}
raw_origins = sorted(df_original['Origin'].unique())

for raw_origin in raw_origins:
    clean_origin = LOCATION_MAPPING.get(raw_origin, raw_origin)
    raw_dests = df_original[df_original['Origin'] == raw_origin]['Destination'].unique().tolist()
    
    if clean_origin not in valid_routes:
        valid_routes[clean_origin] = set()
        
    for raw_dest in raw_dests:
        clean_dest = LOCATION_MAPPING.get(raw_dest, raw_dest)
        if clean_origin != clean_dest:
            valid_routes[clean_origin].add(clean_dest)

valid_routes = {k: sorted(list(v)) for k, v in valid_routes.items()}
print(f"✅ Valid routes: {len(valid_routes)} origins")

# Helper function to get accuracy from different field names
def get_model_accuracy(eval_data):
    """Get accuracy from evaluation data, trying different field names"""
    if not eval_data:
        return 0
    # Try different possible field names
    for field in ['accuracy', 'accuracy_20', 'test_acc_20', 'val_acc_20']:
        if field in eval_data and eval_data[field] > 0:
            return eval_data[field]
    return 0

# Model info for frontend
available_models = {
    'ann': {
        'name': 'Neural Network',
        'description': 'Deep Learning với 3 hidden layers',
        'icon': 'fa-brain',
        'color': '#3b82f6',
        'available': models['ann'] is not None,
        'accuracy': get_model_accuracy(model_evaluation_results.get('ann', {}))
    },
    'linear_regression': {
        'name': 'Linear Regression',
        'description': 'Ridge Regularization (L2)',
        'icon': 'fa-chart-line',
        'color': '#10b981',
        'available': models['linear_regression'] is not None,
        'accuracy': get_model_accuracy(model_evaluation_results.get('linear_regression', {}))
    },
    'decision_tree': {
        'name': 'Decision Tree',
        'description': 'Hyperparameter Tuning',
        'icon': 'fa-sitemap',
        'color': '#f59e0b',
        'available': models['decision_tree'] is not None,
        'accuracy': get_model_accuracy(model_evaluation_results.get('decision_tree', {}))
    }
}

# ============================================================
# AMENITIES ESTIMATION FUNCTIONS
# ============================================================

def estimate_wifi(airline, seat_class, origin, destination):
    intl_airports = ['BKK', 'SIN', 'KUL', 'ICN', 'NRT', 'TPE', 'HKG', 'DXB', 'DOH', 'PVG', 'PEK']
    is_international = origin in intl_airports or destination in intl_airports
    airline_lower = airline.lower()
    
    if seat_class == 'BUSINESS': return 'Yes'
    if is_international and any(x in airline_lower for x in ['vietnam airlines', 'bamboo']): return 'Yes'
    return 'No'

def estimate_meals(airline, seat_class, origin, destination, duration):
    intl_airports = ['BKK', 'SIN', 'KUL', 'ICN', 'NRT', 'TPE', 'HKG', 'DXB', 'DOH', 'PVG', 'PEK']
    is_international = origin in intl_airports or destination in intl_airports
    airline_lower = airline.lower()
    
    if seat_class == 'BUSINESS': return 'Yes'
    if any(x in airline_lower for x in ['vietjet', 'jetstar']): return 'No'
    if is_international: return 'Yes'
    if duration > 120 and any(x in airline_lower for x in ['vietnam airlines', 'bamboo']): return 'Yes'
    return 'No'

def estimate_baggage(airline, seat_class):
    airline_lower = airline.lower()
    if seat_class == 'BUSINESS':
        if any(x in airline_lower for x in ['vietnam airlines', 'bamboo', 'singapore', 'korean']): return 32
        return 30
    else:
        if any(x in airline_lower for x in ['vietnam airlines', 'bamboo']): return 23
        elif any(x in airline_lower for x in ['vietjet', 'jetstar', 'airasia']): return 7
        elif any(x in airline_lower for x in ['singapore', 'thai', 'malaysia', 'korean']): return 23
        elif 'vietravel' in airline_lower: return 20
        return 20

# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================

def create_features(data):
    """Tạo features từ input data"""
    
    airline = data['airline']
    origin = data['origin']
    destination = data['destination']
    day = int(data['day'])
    month = int(data['month'])
    year = int(data['year'])
    departure_hour = int(data['departure_hour'])
    arrival_hour = int(data['arrival_hour'])
    duration = int(data['duration'])
    stops = int(data['stops'])
    seat_class = data['class']
    
    weekday = datetime(year, month, day).weekday()
    time_diff = arrival_hour - departure_hour
    if time_diff < 0: time_diff += 24
    
    # Get route statistics
    route_data = df_original[(df_original['Origin'] == origin) & (df_original['Destination'] == destination)]
    if len(route_data) > 0:
        route_avg_price = route_data['Price_VND'].mean()
        route_std_price = route_data['Price_VND'].std() if len(route_data) > 1 else df_original['Price_VND'].std()
        route_frequency = len(route_data)
    else:
        route_avg_price = df_original['Price_VND'].mean()
        route_std_price = df_original['Price_VND'].std()
        route_frequency = 1
    
    # Get airline statistics
    airline_data = df_original[df_original['Airline'] == airline]
    if len(airline_data) > 0:
        airline_avg_price = airline_data['Price_VND'].mean()
        airline_std_price = airline_data['Price_VND'].std() if len(airline_data) > 1 else df_original['Price_VND'].std()
    else:
        airline_avg_price = df_original['Price_VND'].mean()
        airline_std_price = df_original['Price_VND'].std()
    
    # Get class statistics
    class_data = df_original[df_original['Class'] == seat_class]
    if len(class_data) > 0:
        class_avg_price = class_data['Price_VND'].mean()
    else:
        class_avg_price = df_original['Price_VND'].mean()
    
    features = {}
    
    for feature_name in feature_names:
        if feature_name == 'Airline':
            features[feature_name] = label_encoders['Airline'].transform([airline])[0]
        elif feature_name == 'Origin':
            features[feature_name] = label_encoders['Origin'].transform([origin])[0]
        elif feature_name == 'Destination':
            features[feature_name] = label_encoders['Destination'].transform([destination])[0]
        elif feature_name == 'Day': features[feature_name] = day
        elif feature_name == 'Month': features[feature_name] = month
        elif feature_name == 'Year': features[feature_name] = year
        elif feature_name == 'Weekday': features[feature_name] = weekday
        elif feature_name == 'Departure_Hour': features[feature_name] = departure_hour
        elif feature_name == 'Arrival_Hour': features[feature_name] = arrival_hour
        elif feature_name == 'Duration_Minutes': features[feature_name] = duration
        elif feature_name == 'Stops': features[feature_name] = stops
        elif feature_name == 'Class':
            features[feature_name] = label_encoders['Class'].transform([seat_class])[0]
        elif feature_name == 'Is_Weekend': features[feature_name] = 1 if weekday >= 5 else 0
        elif feature_name == 'Is_Peak': features[feature_name] = 1 if month in [1, 2, 4, 7, 8, 12] else 0
        elif feature_name == 'Hour_Category':
            if 0 <= departure_hour <= 5: features[feature_name] = 0
            elif 6 <= departure_hour <= 11: features[feature_name] = 1
            elif 12 <= departure_hour <= 17: features[feature_name] = 2
            else: features[feature_name] = 3
        elif feature_name == 'Duration_Category':
            if duration < 60: features[feature_name] = 0
            elif duration < 180: features[feature_name] = 1
            else: features[feature_name] = 2
        elif feature_name == 'Route_Avg_Price': features[feature_name] = route_avg_price
        elif feature_name in ['Route_Price_Std', 'Route_Std_Price']: features[feature_name] = route_std_price
        elif feature_name == 'Route_Frequency': features[feature_name] = route_frequency
        elif feature_name == 'Airline_Avg_Price': features[feature_name] = airline_avg_price
        elif feature_name in ['Airline_Price_Std', 'Airline_Std_Price']: features[feature_name] = airline_std_price
        elif feature_name == 'Class_Avg_Price': features[feature_name] = class_avg_price
        elif feature_name == 'Time_Diff': features[feature_name] = time_diff
        elif feature_name == 'Price_Per_Minute': features[feature_name] = route_avg_price / max(duration, 1)
        elif feature_name == 'Route_Time_Interaction': features[feature_name] = route_frequency * duration
        elif feature_name == 'Airline_Route_Interaction': features[feature_name] = airline_avg_price * route_avg_price / 1e6
        elif feature_name == 'Day_Period':
            if day <= 10: features[feature_name] = 0
            elif day <= 20: features[feature_name] = 1
            else: features[feature_name] = 2
        elif feature_name == 'Month_Sin': features[feature_name] = np.sin(2 * np.pi * month / 12)
        elif feature_name == 'Month_Cos': features[feature_name] = np.cos(2 * np.pi * month / 12)
        elif feature_name == 'Weekday_Sin': features[feature_name] = np.sin(2 * np.pi * weekday / 7)
        elif feature_name == 'Weekday_Cos': features[feature_name] = np.cos(2 * np.pi * weekday / 7)
        elif feature_name == 'Hour_Sin': features[feature_name] = np.sin(2 * np.pi * departure_hour / 24)
        elif feature_name == 'Hour_Cos': features[feature_name] = np.cos(2 * np.pi * departure_hour / 24)
        elif feature_name == 'Duration_Hours': features[feature_name] = duration / 60
        elif feature_name == 'Duration_Squared': features[feature_name] = duration ** 2
        elif feature_name == 'Stop_Penalty': features[feature_name] = stops * 0.1
        elif feature_name == 'Has_Stops': features[feature_name] = 1 if stops > 0 else 0
        elif feature_name == 'WiFi':
            wifi = estimate_wifi(airline, seat_class, origin, destination)
            if 'WiFi' in label_encoders: features[feature_name] = label_encoders['WiFi'].transform([wifi])[0]
            else: features[feature_name] = 1 if wifi == 'Yes' else 0
        elif feature_name == 'Meals':
            meals = estimate_meals(airline, seat_class, origin, destination, duration)
            if 'Meals' in label_encoders: features[feature_name] = label_encoders['Meals'].transform([meals])[0]
            else: features[feature_name] = 1 if meals == 'Yes' else 0
        elif feature_name == 'Baggage_kg':
            baggage = estimate_baggage(airline, seat_class)
            features[feature_name] = baggage
        else:
            features[feature_name] = 0
    
    return np.array([features[name] for name in feature_names]).reshape(1, -1)

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_with_ann(features):
    """Dự đoán với ANN model"""
    if models['ann'] is None:
        return None
    features_scaled = scalers['ann'].transform(features)
    prediction = models['ann'].predict(features_scaled, verbose=0)[0][0]
    return float(prediction)

def predict_with_linear_regression(features):
    """Dự đoán với Linear Regression model"""
    if models['linear_regression'] is None:
        return None
    features_scaled = scalers['linear_regression'].transform(features)
    prediction = models['linear_regression'].predict(features_scaled)[0]
    return float(prediction)

def predict_with_decision_tree(features):
    """Dự đoán với Decision Tree model"""
    if models['decision_tree'] is None:
        return None
    # Decision Tree không cần scaling
    prediction = models['decision_tree'].predict(features)[0]
    return float(prediction)

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    """Trang chủ"""
    ui_categories = ['Economy', 'Business']
    valid_airlines = [a for a in unique_values['Airline'] if a != 'Unknown']
    
    clean_origins = sorted(valid_routes.keys())
    
    all_dests = set()
    for dests in valid_routes.values():
        all_dests.update(dests)
    clean_destinations = sorted(list(all_dests))
    
    return render_template('index.html',
                         airlines=sorted(valid_airlines),
                         origins=clean_origins,
                         destinations=clean_destinations,
                         classes=ui_categories,
                         airports=airport_names,
                         accuracy=accuracy,
                         available_models=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    """API dự đoán giá vé - Hỗ trợ nhiều model"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Không nhận được dữ liệu'}), 400
            
        trip_type = data.get('trip_type', 'oneway')
        is_round_trip = trip_type == 'roundtrip'
        model_type = data.get('model', 'ann')  # Default: ANN
        
        # Check if model is available
        if models.get(model_type) is None:
            return jsonify({
                'success': False, 
                'error': f'Model "{model_type}" không khả dụng. Vui lòng chọn model khác.'
            }), 400
        
        # Check if scaler is available (for ANN and Linear Regression)
        if model_type in ['ann', 'linear_regression'] and scalers.get(model_type) is None:
            return jsonify({
                'success': False, 
                'error': f'Scaler cho model "{model_type}" không khả dụng.'
            }), 400
        
        # Check label encoders
        if not label_encoders:
            return jsonify({
                'success': False, 
                'error': 'Label encoders chưa được load. Server cần khởi động lại.'
            }), 500
        
        # Mapping UI Category
        ui_category = data.get('class', 'Economy')
        actual_class = map_category_to_class(ui_category)
        data['class'] = actual_class
        
        # Xử lý tên địa điểm
        input_origin = data.get('origin', '')
        input_dest = data.get('destination', '')
        
        def find_model_label(clean_code, feature_name):
            if feature_name not in label_encoders:
                return clean_code
            if clean_code in label_encoders[feature_name].classes_:
                return clean_code
            potential_matches = [k for k, v in LOCATION_MAPPING.items() if v == clean_code]
            for match in potential_matches:
                if match in label_encoders[feature_name].classes_:
                    return match
            return clean_code

        data['origin'] = find_model_label(input_origin, 'Origin')
        data['destination'] = find_model_label(input_dest, 'Destination')
        
        # Validate inputs
        airline = data.get('airline', '')
        origin = data['origin']
        destination = data['destination']
        seat_class = actual_class
        duration = int(data.get('duration', 120))
        
        errors = []
        if 'Airline' in label_encoders and airline not in label_encoders['Airline'].classes_: 
            errors.append(f'Hãng "{airline}" không có trong dữ liệu')
        if 'Origin' in label_encoders and origin not in label_encoders['Origin'].classes_: 
            errors.append(f'Điểm đi "{origin}" không có trong dữ liệu')
        if 'Destination' in label_encoders and destination not in label_encoders['Destination'].classes_: 
            errors.append(f'Điểm đến "{destination}" không có trong dữ liệu')
        
        if errors:
            return jsonify({'success': False, 'error': ' | '.join(errors)}), 400
        
        # Check if model is available
        if model_type not in models or models[model_type] is None:
            return jsonify({'success': False, 'error': f'Model {model_type} không khả dụng'}), 400
        
        # Create features
        features = create_features(data)
        
        # Predict based on model type
        print(f"\n🔮 Predicting with {model_type.upper()}...")
        
        if model_type == 'ann':
            prediction_raw = predict_with_ann(features)
        elif model_type == 'linear_regression':
            prediction_raw = predict_with_linear_regression(features)
        elif model_type == 'decision_tree':
            prediction_raw = predict_with_decision_tree(features)
        else:
            return jsonify({'success': False, 'error': f'Unknown model type: {model_type}'}), 400
        
        if prediction_raw is None:
            return jsonify({'success': False, 'error': f'Model {model_type} prediction failed'}), 500
        
        # BUSINESS CLASS ADJUSTMENT
        if seat_class == 'BUSINESS':
            prediction = prediction_raw * 2.8
            print(f"   💼 Business class: {prediction_raw:,.0f} → {prediction:,.0f} VNĐ (×2.8)")
        else:
            prediction = prediction_raw
            print(f"   💰 Economy class: {prediction:,.0f} VNĐ")
        
        # Clip giá hợp lý
        prediction = np.clip(prediction, 300000, 25_000_000)
        
        # Round Trip Calculation
        final_price = prediction
        if is_round_trip:
            final_price = prediction * 2.0
            print(f"   ✈️  Round trip: {prediction:,.0f} × 2 = {final_price:,.0f} VNĐ")
        
        # Amenities
        wifi = estimate_wifi(airline, seat_class, input_origin, input_dest)
        meals = estimate_meals(airline, seat_class, input_origin, input_dest, duration)
        baggage_kg = estimate_baggage(airline, seat_class)
        
        # Save History với model type
        try:
            user_ip = request.remote_addr
            user_agent = request.headers.get('User-Agent', 'Unknown')
            search_db.add_search(data, final_price, user_ip, user_agent, wifi, meals, baggage_kg, model_type)
        except Exception as e:
            print(f"   ⚠️ Error saving search: {e}")
        
        # Get model info
        model_info = available_models.get(model_type, {})
        
        return jsonify({
            'success': True,
            'price': float(final_price),
            'price_formatted': f"{final_price:,.0f}".replace(',', '.'),
            'is_round_trip': is_round_trip,
            'model_used': model_type,
            'model_name': model_info.get('name', model_type),
            'model_accuracy': model_info.get('accuracy', 0),
            'amenities': {'wifi': wifi, 'meals': meals, 'baggage_kg': int(baggage_kg)}
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Lỗi: {str(e)}'}), 500

@app.route('/api/valid-routes')
def api_valid_routes():
    """API trả về danh sách routes hợp lệ"""
    return jsonify({'success': True, 'routes': valid_routes})

@app.route('/api/models')
def api_models():
    """API trả về danh sách models khả dụng"""
    return jsonify({'success': True, 'models': available_models})

@app.route('/statistics')
def statistics():
    return render_template('statistics.html', available_models=available_models)

@app.route('/api/statistics')
def api_statistics():
    """API thống kê - Hỗ trợ filter theo model"""
    try:
        model_filter = request.args.get('model', 'all')
        
        # Get search history stats
        search_stats = search_db.get_statistics(model_filter)
        
        # Training data stats
        training_stats = {
            'total_flights': int(len(df_original)),
            'avg_price': float(df_original['Price_VND'].mean()),
            'num_airlines': int(df_original['Airline'].nunique()),
            'num_routes': int(df_original.groupby(['Origin', 'Destination']).ngroups)
        }
        
        # Aggregations
        airline_stats = df_original.groupby('Airline')['Price_VND'].agg(['mean', 'count']).to_dict('index')
        training_stats['airlines'] = {k: {'mean': float(v['mean']), 'count': int(v['count'])} for k, v in airline_stats.items()}
        
        class_stats = df_original.groupby('Class')['Price_VND'].agg(['mean', 'count']).to_dict('index')
        training_stats['classes'] = {k: {'mean': float(v['mean']), 'count': int(v['count'])} for k, v in class_stats.items()}
        
        month_stats = df_original.groupby('Month')['Price_VND'].agg(['mean', 'count']).to_dict('index')
        training_stats['months'] = {int(k): {'mean': float(v['mean']), 'count': int(v['count'])} for k, v in month_stats.items()}
        
        top_routes = df_original.groupby(['Origin', 'Destination'])['Price_VND'].mean().nlargest(10)
        training_stats['top_routes'] = [{'route': f"{o} → {d}", 'mean': float(p)} for (o, d), p in top_routes.items()]
        
        price_ranges = {
            '< 1M': int((df_original['Price_VND'] < 1000000).sum()),
            '1M - 2M': int(((df_original['Price_VND'] >= 1000000) & (df_original['Price_VND'] < 2000000)).sum()),
            '2M - 3M': int(((df_original['Price_VND'] >= 2000000) & (df_original['Price_VND'] < 3000000)).sum()),
            '3M - 5M': int(((df_original['Price_VND'] >= 3000000) & (df_original['Price_VND'] < 5000000)).sum()),
            '> 5M': int((df_original['Price_VND'] >= 5000000).sum())
        }
        training_stats['price_ranges'] = price_ranges
        
        # All model evaluations
        all_model_eval = {}
        for model_name, eval_data in model_evaluation_results.items():
            all_model_eval[model_name] = {
                'accuracy': get_model_accuracy(eval_data),
                'mae': eval_data.get('mae', 0),
                'rmse': eval_data.get('rmse', 0),
                'r2': eval_data.get('r2_score', eval_data.get('r2', 0)),
                'mape': eval_data.get('mape', 0)
            }
        
        return jsonify({
            'success': True,
            'search_history': search_stats,
            'training_data': training_stats,
            'model_evaluation': all_model_eval,
            'available_models': available_models
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug')
def api_debug():
    """Debug endpoint to check model status"""
    return jsonify({
        'success': True,
        'base_dir': BASE_DIR,
        'models_status': {
            'ann': models.get('ann') is not None,
            'linear_regression': models.get('linear_regression') is not None,
            'decision_tree': models.get('decision_tree') is not None
        },
        'scalers_status': {
            'ann': scalers.get('ann') is not None,
            'linear_regression': scalers.get('linear_regression') is not None
        },
        'label_encoders_loaded': len(label_encoders) > 0,
        'feature_names_count': len(feature_names) if feature_names else 0,
        'unique_values_count': len(unique_values) if unique_values else 0,
        'data_rows': len(df_original) if df_original is not None else 0,
        'available_models': available_models
    })

if __name__ == '__main__':
    print("\n🌐 Starting Flask server (Multi-Model)...")
    print("📍 URL: http://localhost:5000")
    print(f"📦 Available models: {[k for k, v in models.items() if v is not None]}")
    app.run(debug=True, host='0.0.0.0', port=5000)