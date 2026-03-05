import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("🚀 FLIGHT PRICE PREDICTION - OPTIMIZED TRAINING")
print("="*80)

# ============================================================
# 1. LOAD AND CLEAN DATA
# ============================================================

print("\n📂 Step 1: Loading data...")
df = pd.read_csv('Flight_Price_Data_Enhanced_Up.csv')
print(f"✅ Loaded: {len(df):,} rows")
print(f"   Price range: {df['Price_VND'].min():,.0f} - {df['Price_VND'].max():,.0f} VNĐ")

# Show data info
print(f"\n📊 Data Overview:")
print(f"   Airlines: {df['Airline'].nunique()}")
print(f"   Origins: {df['Origin'].nunique()}")
print(f"   Destinations: {df['Destination'].nunique()}")
print(f"   Classes: {df['Class'].nunique()}")

# Fix Baggage_kg (convert from "7kg" to 7 if needed)
if df['Baggage_kg'].dtype == 'object':
    print("\n🔧 Converting Baggage_kg from string to numeric...")
    df['Baggage_kg'] = df['Baggage_kg'].astype(str).str.replace('kg', '').astype(float)
    print("✅ Conversion done")

# Convert date columns
df[['Day', 'Month', 'Year']] = df[['Day', 'Month', 'Year']].fillna(0).astype(int)

# Fix Duration
print("\n🔧 Fixing Duration_Minutes...")
df['Duration_Minutes'] = df.apply(
    lambda row: (row['Arrival_Hour'] - row['Departure_Hour']) * 60 
                if row['Arrival_Hour'] >= row['Departure_Hour'] 
                else (row['Arrival_Hour'] + 24 - row['Departure_Hour']) * 60, 
    axis=1
)

# Clean data
print("\n🧹 Cleaning data...")
original_len = len(df)

# Remove invalid durations
df = df[(df['Duration_Minutes'] > 20) & (df['Duration_Minutes'] < 1200)]

# Remove invalid prices
df = df[(df['Price_VND'] > 200000) & (df['Price_VND'] < 10_000_000)]

# Remove extreme stops
df = df[df['Stops'] < 3]

# Remove outliers using IQR
Q1 = df['Price_VND'].quantile(0.10)
Q3 = df['Price_VND'].quantile(0.90)
IQR = Q3 - Q1
df = df[(df['Price_VND'] >= Q1 - 1.5 * IQR) & (df['Price_VND'] <= Q3 + 1.5 * IQR)]

print(f"   Removed: {original_len - len(df):,} rows ({(original_len - len(df))/original_len*100:.1f}%)")
print(f"   Remaining: {len(df):,} rows")
print(f"   New price range: {df['Price_VND'].min():,.0f} - {df['Price_VND'].max():,.0f} VNĐ")

# Fill missing values
df.fillna({
    'Departure_Hour': df['Departure_Hour'].median(),
    'Arrival_Hour': df['Arrival_Hour'].median(),
    'Duration_Minutes': df['Duration_Minutes'].median(),
    'Day': 15,
    'Stops': 0
}, inplace=True)

# ============================================================
# 2. FEATURE ENGINEERING (MATCH APP.PY EXACTLY)
# ============================================================

print("\n🔧 Step 2: Feature engineering...")

# Weekday calculation
df['Weekday'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce').dt.dayofweek
df['Weekday'].fillna(3, inplace=True)  # Default to Wednesday

# Basic features
df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
df['Is_Peak'] = df['Month'].isin([1, 2, 4, 7, 8, 12]).astype(int)

# Hour category
df['Hour_Category'] = pd.cut(df['Departure_Hour'], 
                             bins=[-1, 5, 11, 17, 24], 
                             labels=[0, 1, 2, 3]).fillna(1).astype(int)

# Duration category
df['Duration_Category'] = pd.cut(df['Duration_Minutes'],
                                 bins=[-1, 60, 120, 180, float('inf')],
                                 labels=[0, 1, 2, 3]).fillna(1).astype(int)

# Day period
df['Day_Period'] = pd.cut(df['Day'], bins=[0, 10, 20, 32], labels=[0, 1, 2]).fillna(1).astype(int)

# Cyclical encoding
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Weekday_Sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
df['Weekday_Cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
df['Hour_Sin'] = np.sin(2 * np.pi * df['Departure_Hour'] / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['Departure_Hour'] / 24)

# Time difference
df['Time_Diff'] = (df['Arrival_Hour'] - df['Departure_Hour']).apply(
    lambda x: x if x >= 0 else x + 24)

# Duration features
df['Duration_Hours'] = df['Duration_Minutes'] / 60
df['Duration_Squared'] = df['Duration_Minutes'] ** 2

# Stop features
df['Stop_Penalty'] = df['Stops'] * 0.1
df['Has_Stops'] = (df['Stops'] > 0).astype(int)

print("✅ Basic features created")

# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n🔢 Step 3: Encoding categorical features...")

categorical_cols = ['Airline', 'Origin', 'Destination', 'Class', 'WiFi', 'Meals']
numeric_cols_base = [
    'Day', 'Month', 'Year', 'Weekday', 'Departure_Hour', 'Arrival_Hour',
    'Duration_Minutes', 'Stops', 'Baggage_kg',
    'Hour_Category', 'Duration_Category', 'Is_Weekend', 'Is_Peak',
    'Time_Diff', 'Day_Period',
    'Month_Sin', 'Month_Cos', 'Weekday_Sin', 'Weekday_Cos',
    'Hour_Sin', 'Hour_Cos',
    'Duration_Hours', 'Duration_Squared', 'Stop_Penalty', 'Has_Stops'
]

label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   ✓ {col}: {len(le.classes_)} categories")

# ============================================================
# 4. SPLIT DATA
# ============================================================

print("\n✂️  Step 4: Splitting data...")

feature_cols_base = [col for col in categorical_cols if col in df.columns] + numeric_cols_base
X_base = df[feature_cols_base].copy()
y = df['Price_VND'].values

# Split: 70% train, 15% val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_base, y, test_size=0.15, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42)  # ~15% of total

print(f"✅ Split completed:")
print(f"   Train: {len(X_train):,} ({len(X_train)/len(X_base)*100:.1f}%)")
print(f"   Val:   {len(X_val):,} ({len(X_val)/len(X_base)*100:.1f}%)")
print(f"   Test:  {len(X_test):,} ({len(X_test)/len(X_base)*100:.1f}%)")

# ============================================================
# 5. ADD STATISTICAL FEATURES (TRAIN ONLY)
# ============================================================

print("\n📊 Step 5: Computing statistical features on train set...")

train_df = X_train.copy()
train_df['Price_VND'] = y_train

# Route stats
route_stats = train_df.groupby(['Origin', 'Destination'])['Price_VND'].agg(['mean', 'std', 'count'])

# Airline stats
airline_stats = train_df.groupby('Airline')['Price_VND'].agg(['mean', 'std'])

# Class stats
class_stats = train_df.groupby('Class')['Price_VND'].mean()

global_mean = np.mean(y_train)
global_std = np.std(y_train)

def add_stats_features(df, route_stats, airline_stats, class_stats, global_mean, global_std):
    """Add statistical features"""
    df_merged = df.merge(route_stats, left_on=['Origin', 'Destination'], right_index=True, how='left')
    df_merged.rename(columns={
        'mean': 'Route_Avg_Price',
        'std': 'Route_Price_Std',
        'count': 'Route_Frequency'
    }, inplace=True)
    df_merged.fillna({
        'Route_Avg_Price': global_mean,
        'Route_Price_Std': global_std,
        'Route_Frequency': 1
    }, inplace=True)
    
    df_merged['Airline_Avg_Price'] = df_merged['Airline'].map(airline_stats['mean']).fillna(global_mean)
    df_merged['Airline_Price_Std'] = df_merged['Airline'].map(airline_stats['std']).fillna(global_std)
    df_merged['Class_Avg_Price'] = df_merged['Class'].map(class_stats).fillna(global_mean)
    
    # Interaction features
    df_merged['Price_Per_Minute'] = df_merged['Route_Avg_Price'] / df_merged['Duration_Minutes'].replace(0, 1)
    df_merged['Route_Time_Interaction'] = df_merged['Route_Frequency'] * df_merged['Duration_Minutes']
    df_merged['Airline_Route_Interaction'] = df_merged['Airline_Avg_Price'] * df_merged['Route_Avg_Price'] / 1e6
    
    return df_merged

X_train = add_stats_features(X_train, route_stats, airline_stats, class_stats, global_mean, global_std)
X_val = add_stats_features(X_val, route_stats, airline_stats, class_stats, global_mean, global_std)
X_test = add_stats_features(X_test, route_stats, airline_stats, class_stats, global_mean, global_std)

numeric_cols = numeric_cols_base + [
    'Route_Avg_Price', 'Route_Price_Std', 'Route_Frequency',
    'Airline_Avg_Price', 'Airline_Price_Std', 'Class_Avg_Price',
    'Price_Per_Minute', 'Route_Time_Interaction', 'Airline_Route_Interaction'
]
feature_cols = [col for col in categorical_cols if col in df.columns] + numeric_cols

print(f"✅ Total features: {len(feature_cols)}")

# ============================================================
# 6. SCALE FEATURES
# ============================================================

print("\n⚖️  Step 6: Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[feature_cols])
X_val_scaled = scaler.transform(X_val[feature_cols])
X_test_scaled = scaler.transform(X_test[feature_cols])

print("✅ Scaling done")

# ============================================================
# 7. BUILD MODEL
# ============================================================

print("\n🏗️  Step 7: Building IMPROVED model...")
print("   Improvements:")
print("   ✅ Model capacity: 512→384→256→192→128→64→32→1")
print("   ✅ L2 regularization: 0.001→0.0005")
print("   ✅ Dropout: 0.3→0.25, 0.2→0.15")
print("   ✅ Learning rate: 0.0005→0.001")
print("   ✅ Patience: 50→80, epochs: 500→800")
print("   Expected: +1-2% accuracy improvement")

model = keras.Sequential([
    # Layer 1 - Increased from 256 to 512
    layers.Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],),
                kernel_regularizer=keras.regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    # Layer 2 - New layer
    layers.Dense(384, activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    # Layer 3
    layers.Dense(256, activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Layer 4
    layers.Dense(192, activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Layer 5
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.15),
    
    # Layer 6
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.15),
    
    # Layer 7
    layers.Dense(32, activation='relu'),
    
    # Output
    layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print(f"✅ Model: 512→384→256→192→128→64→32→1")
print(f"   Parameters: {model.count_params():,}")

# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n🚀 Step 8: Training model...")

class AccuracyCallback(callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_acc = 0.0
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0:
            y_pred = self.model.predict(self.X_val, verbose=0).flatten()
            accuracy = np.mean(np.abs(self.y_val - y_pred) / self.y_val <= 0.2) * 100
            self.best_acc = max(self.best_acc, accuracy)
            print(f"\n   Epoch {epoch+1}: Accuracy (±20%) = {accuracy:.2f}% | Best: {self.best_acc:.2f}%")
            
            if accuracy >= 95.0:
                print(f"   🎉 Target reached! Stopping...")
                self.model.stop_training = True

acc_callback = AccuracyCallback(X_val_scaled, y_val)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=80,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    min_lr=0.00001,
    verbose=1
)

os.makedirs('models', exist_ok=True)

checkpoint = callbacks.ModelCheckpoint(
    'models/best_model_improved.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=800,
    batch_size=32,
    callbacks=[early_stop, reduce_lr, acc_callback, checkpoint],
    verbose=2
)

print("\n✅ Training completed!")

# Load best weights
model.load_weights('models/best_model_improved.h5')
print("✅ Loaded best weights")

# ============================================================
# 9. EVALUATE MODEL
# ============================================================

print("\n📊 Step 9: Evaluating model...")

def evaluate_model(X, y, name="Test"):
    y_pred = model.predict(X, verbose=0).flatten()
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    acc_10 = np.mean(np.abs(y - y_pred) / y <= 0.10) * 100
    acc_15 = np.mean(np.abs(y - y_pred) / y <= 0.15) * 100
    acc_20 = np.mean(np.abs(y - y_pred) / y <= 0.20) * 100
    
    print(f"\n{'='*80}")
    print(f"📊 {name.upper()} SET RESULTS")
    print(f"{'='*80}")
    print(f"   MAE:              {mae:,.0f} VNĐ")
    print(f"   RMSE:             {rmse:,.0f} VNĐ")
    print(f"   R² Score:         {r2:.4f}")
    print(f"   MAPE:             {mape:.2f}%")
    print(f"\n   Accuracy (±10%):  {acc_10:.2f}%")
    print(f"   Accuracy (±15%):  {acc_15:.2f}%")
    print(f"   Accuracy (±20%):  {acc_20:.2f}% ⭐")
    
    if acc_20 >= 95:
        print(f"\n   🎉 EXCELLENT! {acc_20:.1f}% >= 95%")
    elif acc_20 >= 90:
        print(f"\n   ✅ VERY GOOD! {acc_20:.1f}% >= 90%")
    elif acc_20 >= 85:
        print(f"\n   ✅ GOOD! {acc_20:.1f}% >= 85%")
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'mape': float(mape),
        'accuracy': float(acc_20),
        'accuracy_10': float(acc_10),
        'accuracy_15': float(acc_15),
        'accuracy_20': float(acc_20)
    }

test_results = evaluate_model(X_test_scaled, y_test, "Test")

# ============================================================
# 10. SAVE MODEL AND ARTIFACTS
# ============================================================

print(f"\n💾 Step 10: Saving model and artifacts...")

# Save model
model.save('models/flight_price_model.h5')
print("✅ Saved: models/flight_price_model.h5")

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: models/scaler.pkl")

# Save label encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✅ Saved: models/label_encoders.pkl")

# Save feature names
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f"✅ Saved: models/feature_names.json ({len(feature_cols)} features)")

# Save unique values for app
unique_values = {col: label_encoders[col].classes_.tolist() for col in label_encoders.keys()}
with open('models/unique_values.json', 'w', encoding='utf-8') as f:
    json.dump(unique_values, f, indent=2, ensure_ascii=False)
print("✅ Saved: models/unique_values.json")

# Save evaluation results
with open('models/evaluation_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)
print("✅ Saved: models/evaluation_results.json")

# Save analytics data for dashboard
analytics_data = {
    "training_samples": len(df),
    "test_samples": len(X_test),
    "accuracy": test_results['accuracy_20'],
    "mae": test_results['mae'],
    "rmse": test_results['rmse'],
    "r2_score": test_results['r2_score'],
    "mape": test_results['mape'],
    "avg_price": float(df['Price_VND'].mean()),
    "features_count": len(feature_cols),
    "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('models/analytics_data.json', 'w', encoding='utf-8') as f:
    json.dump(analytics_data, f, indent=2, ensure_ascii=False)
print("✅ Saved: models/analytics_data.json")

# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "="*80)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\n📊 FINAL RESULTS:")
print(f"   Training Samples:  {len(df):,}")
print(f"   Test Accuracy:     {test_results['accuracy_20']:.2f}%")
print(f"   Test MAE:          {test_results['mae']:,.0f} VNĐ")
print(f"   Test R²:           {test_results['r2_score']:.4f}")

print(f"\n📁 Files saved (ready for app.py):")
print(f"   ✅ models/flight_price_model.h5")
print(f"   ✅ models/scaler.pkl")
print(f"   ✅ models/label_encoders.pkl")
print(f"   ✅ models/feature_names.json")
print(f"   ✅ models/unique_values.json")
print(f"   ✅ models/evaluation_results.json")
print(f"   ✅ models/analytics_data.json")

print(f"\n🚀 Next steps:")
print(f"   1. Copy Flight_Price_Data_Enhanced_Up.csv to app directory")
print(f"   2. Run: python app.py")
print(f"   3. Open: http://localhost:5000")

print("\n" + "="*80)