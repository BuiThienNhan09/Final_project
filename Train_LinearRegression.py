import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 RIDGE REGRESSION MODEL - TRAINING & COMPARISON")
print("="*80)
print("✅ WITH L2 REGULARIZATION & CROSS-VALIDATION (Fair comparison with ANN)")
print("="*80)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n📂 Step 1: Loading data...")
df = pd.read_csv('Flight_Price_Data_Enhanced_Up.csv')
print(f"✅ Loaded: {len(df):,} rows")

# ============================================================
# 2. DATA PREPROCESSING (Same as ANN & Decision Tree)
# ============================================================

print("\n🔧 Step 2: Preprocessing data...")

# Fix Baggage_kg
if df['Baggage_kg'].dtype == 'object':
    df['Baggage_kg'] = df['Baggage_kg'].astype(str).str.replace('kg', '').astype(float)

# Convert date columns
df[['Day', 'Month', 'Year']] = df[['Day', 'Month', 'Year']].fillna(0).astype(int)

# Fix Duration
df['Duration_Minutes'] = df.apply(
    lambda row: (row['Arrival_Hour'] - row['Departure_Hour']) * 60 
                if row['Arrival_Hour'] >= row['Departure_Hour'] 
                else (row['Arrival_Hour'] + 24 - row['Departure_Hour']) * 60, 
    axis=1
)

# Clean data
original_len = len(df)
df = df[(df['Duration_Minutes'] > 20) & (df['Duration_Minutes'] < 1200)]
df = df[(df['Price_VND'] > 200000) & (df['Price_VND'] < 10_000_000)]
df = df[df['Stops'] <= 3]

# Remove outliers
Q1 = df['Price_VND'].quantile(0.10)
Q3 = df['Price_VND'].quantile(0.90)
IQR = Q3 - Q1
df = df[(df['Price_VND'] >= Q1 - 1.5 * IQR) & (df['Price_VND'] <= Q3 + 1.5 * IQR)]

print(f"   Cleaned: {len(df):,} rows (removed {original_len - len(df):,})")

# Fill missing values
df.fillna({
    'Departure_Hour': df['Departure_Hour'].median(),
    'Arrival_Hour': df['Arrival_Hour'].median(),
    'Duration_Minutes': df['Duration_Minutes'].median(),
    'Day': 15,
    'Stops': 0
}, inplace=True)

# ============================================================
# 3. FEATURE ENGINEERING (Same as ANN & Decision Tree)
# ============================================================

print("\n🔧 Step 3: Feature engineering...")

# Weekday
df['Weekday'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce').dt.dayofweek
df['Weekday'].fillna(3, inplace=True)

# Basic features
df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
df['Is_Peak'] = df['Month'].isin([1, 2, 4, 7, 8, 12]).astype(int)
df['Hour_Category'] = pd.cut(df['Departure_Hour'], bins=[-1, 5, 11, 17, 24], labels=[0, 1, 2, 3]).fillna(1).astype(int)
df['Duration_Category'] = pd.cut(df['Duration_Minutes'], bins=[-1, 60, 120, 180, float('inf')], labels=[0, 1, 2, 3]).fillna(1).astype(int)
df['Day_Period'] = pd.cut(df['Day'], bins=[0, 10, 20, 32], labels=[0, 1, 2]).fillna(1).astype(int)

# Cyclical encoding
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Weekday_Sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
df['Weekday_Cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
df['Hour_Sin'] = np.sin(2 * np.pi * df['Departure_Hour'] / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['Departure_Hour'] / 24)

# Time features
df['Time_Diff'] = (df['Arrival_Hour'] - df['Departure_Hour']).apply(lambda x: x if x >= 0 else x + 24)
df['Duration_Hours'] = df['Duration_Minutes'] / 60
df['Duration_Squared'] = df['Duration_Minutes'] ** 2

# Stop features
df['Stop_Penalty'] = df['Stops'] * 0.1
df['Has_Stops'] = (df['Stops'] > 0).astype(int)

print("✅ Feature engineering completed")

# ============================================================
# 4. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n🔢 Step 4: Encoding categorical features...")

categorical_cols = ['Airline', 'Origin', 'Destination', 'Class', 'WiFi', 'Meals']
numeric_cols = [
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
# 5. SPLIT DATA (Same as ANN: 70% train, 15% val, 15% test)
# ============================================================

print("\n✂️  Step 5: Splitting data...")

feature_cols = [col for col in categorical_cols if col in df.columns] + numeric_cols
X = df[feature_cols].copy()
y = df['Price_VND'].values

# Split: 70% train, 15% val, 15% test (Same as ANN & Decision Tree)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42)  # ~15% of total

print(f"✅ Split completed:")
print(f"   Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================
# 6. ADD STATISTICAL FEATURES (Same as ANN & Decision Tree)
# ============================================================

print("\n📊 Step 6: Computing statistical features...")

train_df = X_train.copy()
train_df['Price_VND'] = y_train

route_stats = train_df.groupby(['Origin', 'Destination'])['Price_VND'].agg(['mean', 'std', 'count'])
airline_stats = train_df.groupby('Airline')['Price_VND'].agg(['mean', 'std'])
class_stats = train_df.groupby('Class')['Price_VND'].mean()

global_mean = np.mean(y_train)
global_std = np.std(y_train)

def add_stats_features(df, route_stats, airline_stats, class_stats, global_mean, global_std):
    """Add statistical features (Same as ANN & Decision Tree)"""
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
    
    # Interaction features (Same as ANN & Decision Tree)
    df_merged['Price_Per_Minute'] = df_merged['Route_Avg_Price'] / df_merged['Duration_Minutes'].replace(0, 1)
    df_merged['Route_Time_Interaction'] = df_merged['Route_Frequency'] * df_merged['Duration_Minutes']
    df_merged['Airline_Route_Interaction'] = df_merged['Airline_Avg_Price'] * df_merged['Route_Avg_Price'] / 1e6
    
    return df_merged

X_train = add_stats_features(X_train, route_stats, airline_stats, class_stats, global_mean, global_std)
X_val = add_stats_features(X_val, route_stats, airline_stats, class_stats, global_mean, global_std)
X_test = add_stats_features(X_test, route_stats, airline_stats, class_stats, global_mean, global_std)

print(f"✅ Final features: {X_train.shape[1]}")

# ============================================================
# 7. SCALE FEATURES
# ============================================================

print("\n⚖️  Step 7: Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✅ Scaling completed")

# ============================================================
# 8. TRAIN RIDGE REGRESSION WITH CROSS-VALIDATION
# ============================================================

print("\n🚀 Step 8: Training Ridge Regression with L2 Regularization...")
print("   (Similar to ANN's L2 regularization for fair comparison)")

# Define alphas to try (regularization strength)
# Wider range to find optimal regularization
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

print(f"\n   Testing {len(alphas)} alpha values with 5-fold CV...")
print(f"   Alpha range: {alphas[0]} to {alphas[-1]}")

# Ridge Regression with Cross-Validation
# cv=5 means 5-fold cross-validation (like GridSearchCV in DecisionTree)
ridge_model = RidgeCV(
    alphas=alphas, 
    cv=5,  # 5-fold cross-validation
)

ridge_model.fit(X_train_scaled, y_train)

print(f"\n✅ Training completed!")
print(f"   Best alpha (L2 strength): {ridge_model.alpha_:.4f}")
print(f"   (Lower alpha = less regularization, Higher alpha = more regularization)")

# ============================================================
# 9. EVALUATE MODEL & CHECK FOR OVERFITTING
# ============================================================

print("\n📊 Step 9: Evaluating model and checking for overfitting...")

# Predictions on all sets
y_pred_train = ridge_model.predict(X_train_scaled)
y_pred_val = ridge_model.predict(X_val_scaled)
y_pred_test = ridge_model.predict(X_test_scaled)

# Calculate metrics for TRAIN set
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)
train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
train_acc_10 = np.mean(np.abs(y_train - y_pred_train) / y_train <= 0.10) * 100
train_acc_15 = np.mean(np.abs(y_train - y_pred_train) / y_train <= 0.15) * 100
train_acc_20 = np.mean(np.abs(y_train - y_pred_train) / y_train <= 0.20) * 100

# Calculate metrics for VALIDATION set
val_mae = mean_absolute_error(y_val, y_pred_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
val_r2 = r2_score(y_val, y_pred_val)
val_mape = np.mean(np.abs((y_val - y_pred_val) / y_val)) * 100
val_acc_10 = np.mean(np.abs(y_val - y_pred_val) / y_val <= 0.10) * 100
val_acc_15 = np.mean(np.abs(y_val - y_pred_val) / y_val <= 0.15) * 100
val_acc_20 = np.mean(np.abs(y_val - y_pred_val) / y_val <= 0.20) * 100

# Calculate metrics for TEST set
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
test_acc_10 = np.mean(np.abs(y_test - y_pred_test) / y_test <= 0.10) * 100
test_acc_15 = np.mean(np.abs(y_test - y_pred_test) / y_test <= 0.15) * 100
test_acc_20 = np.mean(np.abs(y_test - y_pred_test) / y_test <= 0.20) * 100

print(f"\n{'='*80}")
print("📊 TRAINING SET RESULTS")
print(f"{'='*80}")
print(f"MAE:                  {train_mae:,.2f} VNĐ")
print(f"RMSE:                 {train_rmse:,.2f} VNĐ")
print(f"R² Score:             {train_r2:.4f}")
print(f"MAPE:                 {train_mape:.2f}%")
print(f"Accuracy (±10%):      {train_acc_10:.2f}%")
print(f"Accuracy (±15%):      {train_acc_15:.2f}%")
print(f"Accuracy (±20%):      {train_acc_20:.2f}%")

print(f"\n{'='*80}")
print("📊 VALIDATION SET RESULTS")
print(f"{'='*80}")
print(f"MAE:                  {val_mae:,.2f} VNĐ")
print(f"RMSE:                 {val_rmse:,.2f} VNĐ")
print(f"R² Score:             {val_r2:.4f}")
print(f"MAPE:                 {val_mape:.2f}%")
print(f"Accuracy (±10%):      {val_acc_10:.2f}%")
print(f"Accuracy (±15%):      {val_acc_15:.2f}%")
print(f"Accuracy (±20%):      {val_acc_20:.2f}%")

print(f"\n{'='*80}")
print("📊 TEST SET RESULTS")
print(f"{'='*80}")
print(f"MAE:                  {test_mae:,.2f} VNĐ")
print(f"RMSE:                 {test_rmse:,.2f} VNĐ")
print(f"R² Score:             {test_r2:.4f}")
print(f"MAPE:                 {test_mape:.2f}%")
print(f"Accuracy (±10%):      {test_acc_10:.2f}%")
print(f"Accuracy (±15%):      {test_acc_15:.2f}%")
print(f"Accuracy (±20%):      {test_acc_20:.2f}%")

# Check for overfitting
print(f"\n{'='*80}")
print("🔍 OVERFITTING ANALYSIS")
print(f"{'='*80}")

overfit_mae_ratio = train_mae / val_mae
overfit_acc_diff = train_acc_20 - val_acc_20

print(f"Train MAE / Val MAE:     {overfit_mae_ratio:.4f}")
print(f"Train Acc - Val Acc:     {overfit_acc_diff:+.2f}%")

if overfit_mae_ratio < 0.90:
    print(f"\n⚠️  SEVERE OVERFITTING DETECTED!")
    print(f"   Training error is much lower than validation error")
    print(f"   Consider: Increase alpha or reduce features")
elif overfit_mae_ratio < 0.95:
    print(f"\n⚠️  Mild overfitting detected")
    print(f"   Model performs slightly better on training data")
elif overfit_mae_ratio > 1.05:
    print(f"\n⚠️  UNDERFITTING DETECTED!")
    print(f"   Model performs worse on training data")
    print(f"   Consider: Decrease alpha or add more features")
else:
    print(f"\n✅ EXCELLENT! Good generalization")
    print(f"   Training and validation performance are similar")
    print(f"   Model is well-regularized (thanks to Ridge L2!)")

# Generalization gap
gen_gap_mae = abs(test_mae - val_mae)
gen_gap_acc = abs(test_acc_20 - val_acc_20)

print(f"\nGeneralization Gap:")
print(f"   |Test MAE - Val MAE|:  {gen_gap_mae:,.2f} VNĐ")
print(f"   |Test Acc - Val Acc|:  {gen_gap_acc:.2f}%")

if gen_gap_mae < val_mae * 0.05:
    print(f"   ✅ Excellent! Test and Val performance are very similar")
elif gen_gap_mae < val_mae * 0.10:
    print(f"   ✅ Good! Small generalization gap")
else:
    print(f"   ⚠️  Large generalization gap - may need more data")

lr_results = {
    'model_type': 'Ridge Regression (L2)',
    'best_alpha': float(ridge_model.alpha_),
    # Test metrics (for comparison with other models)
    'mae': float(test_mae),
    'rmse': float(test_rmse),
    'r2_score': float(test_r2),
    'mape': float(test_mape),
    'accuracy_10': float(test_acc_10),
    'accuracy_15': float(test_acc_15),
    'accuracy_20': float(test_acc_20),
    # Train metrics
    'train_mae': float(train_mae),
    'train_rmse': float(train_rmse),
    'train_r2': float(train_r2),
    'train_acc_20': float(train_acc_20),
    # Val metrics
    'val_mae': float(val_mae),
    'val_rmse': float(val_rmse),
    'val_r2': float(val_r2),
    'val_acc_20': float(val_acc_20),
    # Overfitting metrics
    'overfit_ratio': float(overfit_mae_ratio),
    'overfit_acc_diff': float(overfit_acc_diff),
    'generalization_gap_mae': float(gen_gap_mae),
    'generalization_gap_acc': float(gen_gap_acc)
}

# ============================================================
# 10. LOAD PREVIOUS RESULTS FOR COMPARISON
# ============================================================

print(f"\n{'='*80}")
print("📂 Step 10: Loading previous model results...")
print(f"{'='*80}")

# Load ANN results
ann_results = None
if os.path.exists('models/evaluation_results.json'):
    with open('models/evaluation_results.json', 'r') as f:
        ann_results = json.load(f)
    print("✅ Loaded ANN results")
else:
    print("⚠️  ANN results not found")

# Load Decision Tree results
dt_results = None
if os.path.exists('models/dt_fixed_overfitting_results.json'):
    with open('models/dt_fixed_overfitting_results.json', 'r') as f:
        dt_results = json.load(f)
    print("✅ Loaded Decision Tree results")
else:
    print("⚠️  Decision Tree results not found")

# ============================================================
# 11. CREATE VISUALIZATIONS
# ============================================================

print(f"\n{'='*80}")
print("📊 Step 11: Creating visualizations...")
print(f"{'='*80}")

os.makedirs('model_comparison', exist_ok=True)

# Plot 1: Train vs Val vs Test Comparison (Overfitting Check)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sets = ['Train', 'Validation', 'Test']
mae_values_ridge = [train_mae, val_mae, test_mae]
acc_values_ridge = [train_acc_20, val_acc_20, test_acc_20]
r2_values_ridge = [train_r2, val_r2, test_r2]

# MAE comparison
axes[0].bar(sets, mae_values_ridge, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('MAE (VNĐ)', fontsize=12, fontweight='bold')
axes[0].set_title('Ridge Regression: MAE Across Datasets', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(mae_values_ridge):
    axes[0].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold')

# Accuracy comparison
axes[1].bar(sets, acc_values_ridge, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Ridge Regression: Accuracy (±20%) Across Datasets', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 105])
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(acc_values_ridge):
    axes[1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

# R² comparison
axes[2].bar(sets, r2_values_ridge, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
axes[2].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[2].set_title('Ridge Regression: R² Score Across Datasets', fontsize=14, fontweight='bold')
axes[2].set_ylim([0, 1])
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate(r2_values_ridge):
    axes[2].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison/ridge_overfitting_check.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison/ridge_overfitting_check.png")
plt.close()

# Plot 2: Model Comparison (if other models exist)
if ann_results or dt_results:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = ['Ridge Reg']
    mae_vals = [test_mae]
    rmse_vals = [test_rmse]
    r2_vals = [test_r2]
    mape_vals = [test_mape]
    
    if dt_results:
        models.append('Decision Tree')
        mae_vals.append(dt_results['mae'])
        rmse_vals.append(dt_results['rmse'])
        r2_vals.append(dt_results['r2_score'])
        mape_vals.append(dt_results['mape'])
    
    if ann_results:
        models.append('ANN')
        mae_vals.append(ann_results['mae'])
        rmse_vals.append(ann_results['rmse'])
        r2_vals.append(ann_results['r2_score'])
        mape_vals.append(ann_results['mape'])
    
    colors = ['#3498db', '#27ae60', '#e74c3c'][:len(models)]
    
    # MAE
    axes[0, 0].bar(models, mae_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('MAE (VNĐ)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mae_vals):
        axes[0, 0].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE
    axes[0, 1].bar(models, rmse_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('RMSE (VNĐ)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Root Mean Squared Error Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(rmse_vals):
        axes[0, 1].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # R²
    axes[1, 0].bar(models, r2_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(r2_vals):
        axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MAPE
    axes[1, 1].bar(models, mape_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Mean Absolute Percentage Error Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mape_vals):
        axes[1, 1].text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison/all_models_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison/all_models_comparison.png")
    plt.close()

# Plot 3: Prediction Scatter
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(y_test, y_pred_test, alpha=0.5, s=30, color='#3498db', edgecolors='black', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', lw=3, label='Perfect Prediction')
ax.set_xlabel('Actual Price (VNĐ)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Price (VNĐ)', fontsize=12, fontweight='bold')
ax.set_title('Ridge Regression: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

textstr = f'R² = {test_r2:.4f}\nMAE = {test_mae:,.0f} VNĐ\nAlpha = {ridge_model.alpha_:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('model_comparison/ridge_prediction_scatter.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison/ridge_prediction_scatter.png")
plt.close()

# ============================================================
# PHẦN SO SÁNH CHI TIẾT 3 MÔ HÌNH
# ============================================================

if ann_results and dt_results:
    print(f"\n{'='*80}")
    print("📊 TẠO BIỂU ĐỒ SO SÁNH CHI TIẾT 3 MÔ HÌNH")
    print(f"{'='*80}")
    
    # Chuẩn bị dữ liệu cho 3 mô hình
    models_names = ['Linear\nRegression', 'Decision\nTree', 'ANN']
    models_names_short = ['Linear Reg', 'Decision Tree', 'ANN']
    
    # Test metrics
    mae_comparison = [test_mae, dt_results['mae'], ann_results['mae']]
    rmse_comparison = [test_rmse, dt_results['rmse'], ann_results['rmse']]
    r2_comparison = [test_r2, dt_results['r2_score'], ann_results['r2_score']]
    mape_comparison = [test_mape, dt_results['mape'], ann_results['mape']]
    acc_10_comparison = [test_acc_10, dt_results.get('accuracy_10', 0), ann_results.get('accuracy_10', 0)]
    acc_15_comparison = [test_acc_15, dt_results.get('accuracy_15', 0), ann_results.get('accuracy_15', 0)]
    acc_20_comparison = [test_acc_20, dt_results.get('accuracy_20', dt_results.get('accuracy', 0)), 
                         ann_results.get('accuracy_20', ann_results.get('accuracy', 0))]
    
    colors_3models = ['#3498db', '#27ae60', '#e74c3c']
    
    # ============================================================
    # Plot 4: Biểu đồ cột so sánh tổng quan 4 metrics chính
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # MAE Comparison
    bars_mae = axes[0, 0].bar(models_names, mae_comparison, color=colors_3models, 
                               alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('MAE (VNĐ)', fontsize=14, fontweight='bold')
    axes[0, 0].set_title('So Sánh Mean Absolute Error (MAE)', fontsize=16, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars_mae, mae_comparison)):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:,.0f}\nVNĐ',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # RMSE Comparison
    bars_rmse = axes[0, 1].bar(models_names, rmse_comparison, color=colors_3models, 
                                alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('RMSE (VNĐ)', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('So Sánh Root Mean Squared Error (RMSE)', fontsize=16, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars_rmse, rmse_comparison)):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:,.0f}\nVNĐ',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # R² Score Comparison
    bars_r2 = axes[1, 0].bar(models_names, r2_comparison, color=colors_3models, 
                              alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('R² Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_title('So Sánh R² Score (Hệ Số Xác Định)', fontsize=16, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1, 0].axhline(y=0.9, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Mục tiêu: 0.90')
    axes[1, 0].legend(loc='lower right', fontsize=10)
    for i, (bar, val) in enumerate(zip(bars_r2, r2_comparison)):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # MAPE Comparison
    bars_mape = axes[1, 1].bar(models_names, mape_comparison, color=colors_3models, 
                                alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('MAPE (%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_title('So Sánh Mean Absolute Percentage Error (MAPE)', fontsize=16, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars_mape, mape_comparison)):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}%',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.suptitle('SO SÁNH HIỆU SUẤT 3 MÔ HÌNH DỰ ĐOÁN GIÁ VÉ MÁY BAY', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('model_comparison/detailed_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison/detailed_metrics_comparison.png")
    plt.close()
    
    # ============================================================
    # Plot 5: So sánh độ chính xác ở các ngưỡng khác nhau
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(models_names_short))
    width = 0.25
    
    bars1 = ax.bar(x_pos - width, acc_10_comparison, width, label='±10%', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos, acc_15_comparison, width, label='±15%', 
                   color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x_pos + width, acc_20_comparison, width, label='±20%', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Mô Hình', fontsize=14, fontweight='bold')
    ax.set_ylabel('Độ Chính Xác (%)', fontsize=14, fontweight='bold')
    ax.set_title('So Sánh Độ Chính Xác Ở Các Ngưỡng Sai Số', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_names_short, fontsize=12)
    ax.legend(fontsize=12, title='Ngưỡng Sai Số', title_fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Mục tiêu: 95%')
    
    # Thêm giá trị lên các cột
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison/accuracy_levels_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison/accuracy_levels_comparison.png")
    plt.close()
    
    # ============================================================
    # Plot 6: Biểu đồ tròn (Pie Chart) - Phân bố độ chính xác
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (model_name, acc_20) in enumerate(zip(models_names_short, acc_20_comparison)):
        # Tính toán phần trăm dự đoán chính xác và không chính xác
        accurate = acc_20
        inaccurate = 100 - acc_20
        
        sizes = [accurate, inaccurate]
        labels = [f'Chính xác (±20%)\n{accurate:.2f}%', f'Không chính xác\n{inaccurate:.2f}%']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        
        axes[idx].pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                     autopct='%1.1f%%', shadow=True, startangle=90,
                     textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[idx].set_title(f'{model_name}\nĐộ Chính Xác: {acc_20:.2f}%', 
                           fontsize=14, fontweight='bold')
    
    plt.suptitle('PHÂN BỐ ĐỘ CHÍNH XÁC CỦA 3 MÔ HÌNH (±20%)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_comparison/accuracy_pie_charts.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison/accuracy_pie_charts.png")
    plt.close()
    
    # ============================================================
    # Plot 7: Radar Chart - So sánh đa chiều
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Chuẩn hóa các metrics về thang 0-100 để vẽ radar chart
    categories = ['Accuracy\n(±20%)', 'R² Score\n(x100)', 'MAE\n(Inverted)', 
                  'RMSE\n(Inverted)', 'MAPE\n(Inverted)']
    N = len(categories)
    
    # Tính giá trị đã chuẩn hóa cho từng mô hình
    def normalize_metrics(acc, r2, mae, rmse, mape):
        # Accuracy và R² càng cao càng tốt (đã ở dạng %)
        # MAE, RMSE, MAPE càng thấp càng tốt -> đảo ngược
        mae_norm = 100 - min(100, (mae / 500000) * 100)  # Scale MAE
        rmse_norm = 100 - min(100, (rmse / 600000) * 100)  # Scale RMSE
        mape_norm = 100 - min(100, mape)  # MAPE đã ở dạng %
        return [acc, r2 * 100, mae_norm, rmse_norm, mape_norm]
    
    linear_values = normalize_metrics(acc_20_comparison[0], r2_comparison[0], 
                                     mae_comparison[0], rmse_comparison[0], mape_comparison[0])
    dt_values = normalize_metrics(acc_20_comparison[1], r2_comparison[1], 
                                  mae_comparison[1], rmse_comparison[1], mape_comparison[1])
    ann_values = normalize_metrics(acc_20_comparison[2], r2_comparison[2], 
                                   mae_comparison[2], rmse_comparison[2], mape_comparison[2])
    
    # Thêm điểm đầu tiên vào cuối để đóng vòng
    linear_values += linear_values[:1]
    dt_values += dt_values[:1]
    ann_values += ann_values[:1]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Vẽ radar chart
    ax.plot(angles, linear_values, 'o-', linewidth=2.5, label='Linear Regression', color='#3498db')
    ax.fill(angles, linear_values, alpha=0.15, color='#3498db')
    
    ax.plot(angles, dt_values, 'o-', linewidth=2.5, label='Decision Tree', color='#27ae60')
    ax.fill(angles, dt_values, alpha=0.15, color='#27ae60')
    
    ax.plot(angles, ann_values, 'o-', linewidth=2.5, label='ANN', color='#e74c3c')
    ax.fill(angles, ann_values, alpha=0.15, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_title('SO SÁNH ĐA CHIỀU HIỆU SUẤT 3 MÔ HÌNH\n(Giá trị càng gần 100 càng tốt)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('model_comparison/radar_chart_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison/radar_chart_comparison.png")
    plt.close()
    
    # ============================================================
    # Plot 8: Bảng tổng hợp so sánh (Table Visualization)
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Tạo dữ liệu bảng
    table_data = [
        ['Metrics', 'Linear Regression', 'Decision Tree', 'ANN', 'Mô Hình Tốt Nhất'],
        ['MAE (VNĐ)', f'{mae_comparison[0]:,.0f}', f'{mae_comparison[1]:,.0f}', 
         f'{mae_comparison[2]:,.0f}', models_names_short[np.argmin(mae_comparison)]],
        ['RMSE (VNĐ)', f'{rmse_comparison[0]:,.0f}', f'{rmse_comparison[1]:,.0f}', 
         f'{rmse_comparison[2]:,.0f}', models_names_short[np.argmin(rmse_comparison)]],
        ['R² Score', f'{r2_comparison[0]:.4f}', f'{r2_comparison[1]:.4f}', 
         f'{r2_comparison[2]:.4f}', models_names_short[np.argmax(r2_comparison)]],
        ['MAPE (%)', f'{mape_comparison[0]:.2f}', f'{mape_comparison[1]:.2f}', 
         f'{mape_comparison[2]:.2f}', models_names_short[np.argmin(mape_comparison)]],
        ['Accuracy ±10% (%)', f'{acc_10_comparison[0]:.2f}', f'{acc_10_comparison[1]:.2f}', 
         f'{acc_10_comparison[2]:.2f}', models_names_short[np.argmax(acc_10_comparison)]],
        ['Accuracy ±15% (%)', f'{acc_15_comparison[0]:.2f}', f'{acc_15_comparison[1]:.2f}', 
         f'{acc_15_comparison[2]:.2f}', models_names_short[np.argmax(acc_15_comparison)]],
        ['Accuracy ±20% (%)', f'{acc_20_comparison[0]:.2f}', f'{acc_20_comparison[1]:.2f}', 
         f'{acc_20_comparison[2]:.2f}', models_names_short[np.argmax(acc_20_comparison)]],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Định dạng header
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Định dạng các hàng
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('#ffffff')
            
            # Highlight cột mô hình tốt nhất
            if j == 4:
                cell.set_facecolor('#2ecc71')
                cell.set_text_props(weight='bold', color='white')
    
    plt.title('BẢNG TỔNG HỢP SO SÁNH CHI TIẾT 3 MÔ HÌNH', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig('model_comparison/comparison_table.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison/comparison_table.png")
    plt.close()
    
    # ============================================================
    # Plot 9: Xếp hạng tổng thể các mô hình
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Tính điểm tổng hợp cho mỗi mô hình (càng cao càng tốt)
    def calculate_score(acc, r2, mae, rmse, mape):
        # Normalize về thang 0-100
        acc_score = acc  # Đã ở dạng %
        r2_score = r2 * 100  # Chuyển về %
        mae_score = max(0, 100 - (mae / 500000) * 100)
        rmse_score = max(0, 100 - (rmse / 600000) * 100)
        mape_score = max(0, 100 - mape)
        
        # Trọng số: Accuracy (30%), R² (25%), MAE (20%), RMSE (15%), MAPE (10%)
        total_score = (acc_score * 0.30 + r2_score * 0.25 + mae_score * 0.20 + 
                      rmse_score * 0.15 + mape_score * 0.10)
        return total_score
    
    linear_score = calculate_score(acc_20_comparison[0], r2_comparison[0], 
                                   mae_comparison[0], rmse_comparison[0], mape_comparison[0])
    dt_score = calculate_score(acc_20_comparison[1], r2_comparison[1], 
                               mae_comparison[1], rmse_comparison[1], mape_comparison[1])
    ann_score = calculate_score(acc_20_comparison[2], r2_comparison[2], 
                                mae_comparison[2], rmse_comparison[2], mape_comparison[2])
    
    scores = [linear_score, dt_score, ann_score]
    
    # Sắp xếp mô hình theo điểm
    model_score_pairs = list(zip(models_names_short, scores, colors_3models))
    model_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    sorted_models = [x[0] for x in model_score_pairs]
    sorted_scores = [x[1] for x in model_score_pairs]
    sorted_colors = [x[2] for x in model_score_pairs]
    
    bars = ax.barh(sorted_models, sorted_scores, color=sorted_colors, 
                   alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xlabel('Điểm Tổng Hợp (0-100)', fontsize=14, fontweight='bold')
    ax.set_title('XẾP HẠNG TỔNG THỂ CÁC MÔ HÌNH\n(Dựa trên tổng hợp tất cả metrics)', 
                fontsize=16, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Thêm giá trị và xếp hạng
    for idx, (bar, score, model) in enumerate(zip(bars, sorted_scores, sorted_models)):
        width = bar.get_width()
        rank = idx + 1
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉'
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {medal} #{rank}: {score:.2f} điểm',
               ha='left', va='center', fontweight='bold', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('model_comparison/overall_ranking.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison/overall_ranking.png")
    plt.close()
    
    # ============================================================
    # Tạo file JSON tổng hợp kết quả so sánh
    # ============================================================
    comparison_summary = {
        'models': {
            'linear_regression': {
                'mae': float(mae_comparison[0]),
                'rmse': float(rmse_comparison[0]),
                'r2_score': float(r2_comparison[0]),
                'mape': float(mape_comparison[0]),
                'accuracy_10': float(acc_10_comparison[0]),
                'accuracy_15': float(acc_15_comparison[0]),
                'accuracy_20': float(acc_20_comparison[0]),
                'overall_score': float(linear_score)
            },
            'decision_tree': {
                'mae': float(mae_comparison[1]),
                'rmse': float(rmse_comparison[1]),
                'r2_score': float(r2_comparison[1]),
                'mape': float(mape_comparison[1]),
                'accuracy_10': float(acc_10_comparison[1]),
                'accuracy_15': float(acc_15_comparison[1]),
                'accuracy_20': float(acc_20_comparison[1]),
                'overall_score': float(dt_score)
            },
            'ann': {
                'mae': float(mae_comparison[2]),
                'rmse': float(rmse_comparison[2]),
                'r2_score': float(r2_comparison[2]),
                'mape': float(mape_comparison[2]),
                'accuracy_10': float(acc_10_comparison[2]),
                'accuracy_15': float(acc_15_comparison[2]),
                'accuracy_20': float(acc_20_comparison[2]),
                'overall_score': float(ann_score)
            }
        },
        'best_model_per_metric': {
            'lowest_mae': models_names_short[np.argmin(mae_comparison)],
            'lowest_rmse': models_names_short[np.argmin(rmse_comparison)],
            'highest_r2': models_names_short[np.argmax(r2_comparison)],
            'lowest_mape': models_names_short[np.argmin(mape_comparison)],
            'highest_accuracy': models_names_short[np.argmax(acc_20_comparison)]
        },
        'overall_ranking': [
            {'rank': i+1, 'model': model, 'score': float(score)} 
            for i, (model, score) in enumerate(zip(sorted_models, sorted_scores))
        ],
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('model_comparison/comparison_summary.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
    print("✅ Saved: model_comparison/comparison_summary.json")
    
    print(f"\n{'='*80}")
    print("✅ ĐÃ TẠO XONG TẤT CẢ CÁC BIỂU ĐỒ SO SÁNH!")
    print(f"{'='*80}")
    print(f"\n📊 Các file biểu đồ đã tạo:")
    print(f"   1. detailed_metrics_comparison.png - So sánh 4 metrics chính")
    print(f"   2. accuracy_levels_comparison.png - So sánh độ chính xác ở các ngưỡng")
    print(f"   3. accuracy_pie_charts.png - Biểu đồ tròn phân bố độ chính xác")
    print(f"   4. radar_chart_comparison.png - Biểu đồ radar đa chiều")
    print(f"   5. comparison_table.png - Bảng tổng hợp chi tiết")
    print(f"   6. overall_ranking.png - Xếp hạng tổng thể")
    print(f"   7. comparison_summary.json - File JSON tổng hợp kết quả")
    
    print(f"\n🏆 XẾP HẠNG TỔNG THỂ:")
    for i, (model, score) in enumerate(zip(sorted_models, sorted_scores)):
        medal = '🥇' if i == 0 else '🥈' if i == 1 else '🥉'
        print(f"   {medal} #{i+1}: {model:20s} - {score:.2f} điểm")
    
else:
    print(f"\n⚠️  Không thể tạo biểu đồ so sánh chi tiết - thiếu kết quả từ ANN hoặc Decision Tree")
    print(f"   Vui lòng chạy Train_ANN.py và Train_DecisionTree.py trước")


# ============================================================
# 12. SAVE MODEL AND RESULTS
# ============================================================

print(f"\n{'='*80}")
print("💾 Step 12: Saving model and results...")
print(f"{'='*80}")

os.makedirs('models', exist_ok=True)

# Save Ridge model
with open('models/ridge_regression_model.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)
print("✅ Saved: models/ridge_regression_model.pkl")

# Save scaler
with open('models/ridge_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: models/ridge_scaler.pkl")

# Save results
with open('models/ridge_evaluation_results.json', 'w') as f:
    json.dump(lr_results, f, indent=2)
print("✅ Saved: models/ridge_evaluation_results.json")

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print(f"\n{'='*80}")
print("✅ RIDGE REGRESSION TRAINING COMPLETED!")
print(f"{'='*80}")

print(f"\n📊 KEY RESULTS:")
print(f"   Best Alpha (L2):      {ridge_model.alpha_:.4f}")
print(f"   Test MAE:             {test_mae:,.0f} VNĐ")
print(f"   Test Accuracy (±20%): {test_acc_20:.2f}%")
print(f"   Test R²:              {test_r2:.4f}")
print(f"   Overfit Ratio:        {overfit_mae_ratio:.4f} {'✅' if 0.95 <= overfit_mae_ratio <= 1.05 else '⚠️'}")

print(f"\n✅ ADVANTAGES OVER BASIC LINEAR REGRESSION:")
print(f"   • L2 Regularization prevents overfitting")
print(f"   • Cross-validation finds optimal alpha automatically")
print(f"   • Fair comparison with ANN (both have L2)")
print(f"   • Validation monitoring like ANN")

print(f"\n{'='*80}")