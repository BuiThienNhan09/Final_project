import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
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
print("🌳 DECISION TREE MODEL - TRAINING & COMPARISON")
print("="*80)
print("✅ WITH HYPERPARAMETER TUNING & VALIDATION TRACKING (Fair comparison with ANN)")
print("="*80)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n📂 Step 1: Loading data...")
df = pd.read_csv('Flight_Price_Data_Enhanced_Up.csv')
print(f"✅ Loaded: {len(df):,} rows")

# ============================================================
# 2. DATA PREPROCESSING (Same as ANN & Ridge Regression)
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
# 3. FEATURE ENGINEERING (Same as ANN & Ridge Regression)
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

# Split: 70% train, 15% val, 15% test (Same as ANN & Ridge)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42)  # ~15% of total

print(f"✅ Split completed:")
print(f"   Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================
# 6. ADD STATISTICAL FEATURES (Same as ANN & Ridge)
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
    """Add statistical features (Same as ANN & Ridge)"""
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
    
    # Interaction features (Same as ANN & Ridge)
    df_merged['Price_Per_Minute'] = df_merged['Route_Avg_Price'] / df_merged['Duration_Minutes'].replace(0, 1)
    df_merged['Route_Time_Interaction'] = df_merged['Route_Frequency'] * df_merged['Duration_Minutes']
    df_merged['Airline_Route_Interaction'] = df_merged['Airline_Avg_Price'] * df_merged['Route_Avg_Price'] / 1e6
    
    return df_merged

X_train = add_stats_features(X_train, route_stats, airline_stats, class_stats, global_mean, global_std)
X_val = add_stats_features(X_val, route_stats, airline_stats, class_stats, global_mean, global_std)
X_test = add_stats_features(X_test, route_stats, airline_stats, class_stats, global_mean, global_std)

print(f"✅ Final features: {X_train.shape[1]}")

# NOTE: Decision Tree does NOT need scaling (unlike Ridge and ANN)
print("\n⚠️  Note: Decision Tree does NOT require feature scaling")

# ============================================================
# 7. TRAIN DECISION TREE WITH HYPERPARAMETER TUNING
# ============================================================

print("\n🚀 Step 7: Training Decision Tree with ANTI-OVERFITTING constraints...")
print("   (Fixed from previous version that had severe overfitting)")
print("   Changes to PREVENT overfitting:")
print("   • Reduced max_depth: [10, 15, 20] (was None/25/30)")
print("   • Increased min_samples_split: [20, 40, 60] (was 2/5/10)")
print("   • Increased min_samples_leaf: [10, 20, 30] (was 1/2/4)")
print("   • Added stronger pruning: min_impurity_decrease [0.01, 0.05, 0.1]")
print("   Expected: Test accuracy may drop but production will be STABLE!")

# Define parameter grid - MUCH MORE RESTRICTIVE to prevent overfitting
# Previous grid allowed tree depth 34 with 5195 leaves - SEVERE OVERFITTING!
# New grid forces simpler, more generalizable trees
param_grid = {
    'max_depth': [10, 15, 20],  # Reduced from [10, 15, 20, 25, 30, None]
    'min_samples_split': [20, 40, 60],  # Increased from [2, 5, 10, 20]
    'min_samples_leaf': [10, 20, 30],  # Increased from [1, 2, 4, 8]
    'max_features': ['sqrt', 'log2'],  # Removed None option
    'min_impurity_decrease': [0.01, 0.05, 0.1]  # Stronger pruning than [0.0, 0.001, 0.01]
}

print(f"\n   Parameter grid size: {len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) * len(param_grid['min_impurity_decrease'])} combinations")
print(f"   Using 5-fold cross-validation")
print(f"\n   Key constraints:")
print(f"   • Max tree depth: 20 (prevents deep memorization)")
print(f"   • Min samples to split: 20-60 (prevents tiny splits)")
print(f"   • Min samples per leaf: 10-30 (prevents overfitting to outliers)")
print(f"   • Stronger pruning: 0.01-0.1 (removes weak splits)")

# Base model
dt_base = DecisionTreeRegressor(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=dt_base,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation (same as Ridge)
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

print(f"\n   Starting GridSearchCV (this may take several minutes)...")
grid_search.fit(X_train, y_train)

# Best model
dt_model = grid_search.best_estimator_

print(f"\n✅ Training completed!")
print(f"\n   Best hyperparameters found:")
for param, value in grid_search.best_params_.items():
    print(f"      {param:25s}: {value}")

print(f"\n   Best CV score (MAE):        {-grid_search.best_score_:,.2f} VNĐ")

# ============================================================
# 8. EVALUATE MODEL & CHECK FOR OVERFITTING
# ============================================================

print(f"\n{'='*80}")
print("📊 Step 8: Evaluating model and checking for overfitting...")
print(f"{'='*80}")

# Predictions on all sets
y_pred_train = dt_model.predict(X_train)
y_pred_val = dt_model.predict(X_val)
y_pred_test = dt_model.predict(X_test)

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
overfit_r2_diff = train_r2 - val_r2
overfit_acc_diff = train_acc_20 - val_acc_20

print(f"Train MAE / Val MAE:     {overfit_mae_ratio:.4f}")
print(f"Train R² - Val R²:       {overfit_r2_diff:+.4f}")
print(f"Train Acc - Val Acc:     {overfit_acc_diff:+.2f}%")

if overfit_mae_ratio < 0.85:
    print(f"\n⚠️  SEVERE OVERFITTING DETECTED!")
    print(f"   Training error is MUCH lower than validation error")
    print(f"   Tree is memorizing training data")
    print(f"   Consider: Reduce max_depth, increase min_samples_split/leaf")
elif overfit_mae_ratio < 0.90:
    print(f"\n⚠️  Moderate overfitting detected")
    print(f"   Model performs notably better on training data")
    print(f"   GridSearch should help mitigate this")
elif overfit_mae_ratio < 0.95:
    print(f"\n⚠️  Mild overfitting detected")
    print(f"   Some performance gap between train and validation")
elif overfit_mae_ratio > 1.05:
    print(f"\n⚠️  UNDERFITTING DETECTED!")
    print(f"   Model performs worse on training data (unusual for trees)")
    print(f"   Consider: Increase max_depth, decrease min_samples constraints")
else:
    print(f"\n✅ EXCELLENT! Good generalization")
    print(f"   Training and validation performance are similar")
    print(f"   Hyperparameter tuning successfully prevented overfitting!")

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

# Tree complexity
print(f"\n{'='*80}")
print("🌳 TREE COMPLEXITY")
print(f"{'='*80}")
print(f"Tree depth:              {dt_model.get_depth()}")
print(f"Number of leaves:        {dt_model.get_n_leaves()}")
print(f"Total nodes:             {dt_model.tree_.node_count}")

if dt_model.get_depth() > 30:
    print(f"   ⚠️  Very deep tree - high risk of overfitting")
elif dt_model.get_depth() > 20:
    print(f"   ⚠️  Deep tree - monitor for overfitting")
else:
    print(f"   ✅ Reasonable tree depth")

dt_results = {
    'model_type': 'Decision Tree (Tuned)',
    'best_params': grid_search.best_params_,
    'tree_depth': int(dt_model.get_depth()),
    'n_leaves': int(dt_model.get_n_leaves()),
    # Test metrics (for comparison)
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
    'overfit_r2_diff': float(overfit_r2_diff),
    'overfit_acc_diff': float(overfit_acc_diff),
    'generalization_gap_mae': float(gen_gap_mae),
    'generalization_gap_acc': float(gen_gap_acc)
}

# ============================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ============================================================

print(f"\n{'='*80}")
print("📊 Step 9: Analyzing feature importance...")
print(f"{'='*80}")

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.4f}")

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

# Load Ridge results
ridge_results = None
if os.path.exists('models/ridge_evaluation_results.json'):
    with open('models/ridge_evaluation_results.json', 'r') as f:
        ridge_results = json.load(f)
    print("✅ Loaded Ridge Regression results")
else:
    print("⚠️  Ridge Regression results not found")

# ============================================================
# 11. CREATE VISUALIZATIONS
# ============================================================

print(f"\n{'='*80}")
print("📊 Step 11: Creating visualizations...")
print(f"{'='*80}")

os.makedirs('model_comparison', exist_ok=True)

# Plot 1: Train vs Val vs Test (Overfitting Check)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sets = ['Train', 'Validation', 'Test']
mae_values_dt = [train_mae, val_mae, test_mae]
acc_values_dt = [train_acc_20, val_acc_20, test_acc_20]
r2_values_dt = [train_r2, val_r2, test_r2]

# MAE
axes[0].bar(sets, mae_values_dt, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('MAE (VNĐ)', fontsize=12, fontweight='bold')
axes[0].set_title('Decision Tree: MAE Across Datasets', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(mae_values_dt):
    axes[0].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold')

# Accuracy
axes[1].bar(sets, acc_values_dt, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Decision Tree: Accuracy (±20%) Across Datasets', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 105])
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(acc_values_dt):
    axes[1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

# R²
axes[2].bar(sets, r2_values_dt, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
axes[2].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[2].set_title('Decision Tree: R² Score Across Datasets', fontsize=14, fontweight='bold')
axes[2].set_ylim([0, 1])
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate(r2_values_dt):
    axes[2].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison/dt_fixed_overfitting_check.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison/dt_fixed_overfitting_check.png")
plt.close()

# Plot 2: Feature Importance
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance.head(20)
bars = ax.barh(range(len(top_features)), top_features['importance'], 
               color='#27ae60', alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Decision Tree: Top 20 Feature Importances', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
    ax.text(val, i, f' {val:.4f}', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison/dt_feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison/dt_feature_importance.png")
plt.close()

# Plot 3: Prediction Scatter
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(y_test, y_pred_test, alpha=0.5, s=30, color='#27ae60', edgecolors='black', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', lw=3, label='Perfect Prediction')
ax.set_xlabel('Actual Price (VNĐ)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Price (VNĐ)', fontsize=12, fontweight='bold')
ax.set_title('Decision Tree: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

textstr = f'R² = {test_r2:.4f}\nMAE = {test_mae:,.0f} VNĐ\nDepth = {dt_model.get_depth()}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('model_comparison/dt_prediction_scatter.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison/dt_prediction_scatter.png")
plt.close()

# ============================================================
# 12. SAVE MODEL AND RESULTS
# ============================================================

print(f"\n{'='*80}")
print("💾 Step 12: Saving model and results...")
print(f"{'='*80}")

os.makedirs('models', exist_ok=True)

# Save model
with open('models/decision_tree_fixed_overfitting.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
print("✅ Saved: models/decision_tree_fixed_overfitting.pkl")

# Save results
with open('models/dt_fixed_overfitting_results.json', 'w') as f:
    json.dump(dt_results, f, indent=2)
print("✅ Saved: models/dt_fixed_overfitting_results.json")

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print(f"\n{'='*80}")
print("✅ DECISION TREE TRAINING COMPLETED!")
print(f"{'='*80}")

print(f"\n📊 KEY RESULTS:")
print(f"   Test MAE:             {test_mae:,.0f} VNĐ")
print(f"   Test Accuracy (±20%): {test_acc_20:.2f}%")
print(f"   Test R²:              {test_r2:.4f}")
print(f"   Tree Depth:           {dt_model.get_depth()}")
print(f"   Overfit Ratio:        {overfit_mae_ratio:.4f} {'✅' if overfit_mae_ratio >= 0.90 else '⚠️'}")

print(f"\n✅ ADVANTAGES:")
print(f"   • Hyperparameter tuning with GridSearchCV (5-fold CV)")
print(f"   • Validation tracking detects overfitting")
print(f"   • Fair comparison with ANN and Ridge")
print(f"   • Feature importance insights")
print(f"   • No scaling required")

print(f"\n{'='*80}")