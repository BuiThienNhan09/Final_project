"""
EDA - EXPLORATORY DATA ANALYSIS (PHIÊN BẢN CẢI TIẾN)
Phân Tích Khám Phá Dữ Liệu Chuyên Nghiệp
Tác giả: Bui Thien Nhan
Đã sửa lỗi và cải tiến cho dữ liệu Flight Price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Cấu hình matplotlib để hỗ trợ tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FlightDataEDA:
    """Class EDA chuyên nghiệp cho dữ liệu vé máy bay"""
    
    def __init__(self, data_path, output_dir='eda_plots'):
        """
        Khởi tạo EDA
        
        Args:
            data_path: Đường dẫn file CSV
            output_dir: Thư mục lưu biểu đồ
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        # Tạo thư mục output
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Đã tạo thư mục: {output_dir}")
    
    def load_data(self):
        """Đọc dữ liệu"""
        print("\n" + "="*70)
        print("📂 BƯỚC 1: ĐỌC DỮ LIỆU")
        print("="*70)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"✓ Đã đọc: {len(self.df):,} dòng × {len(self.df.columns)} cột")
        print(f"✓ Kích thước: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.df
    
    def basic_info(self):
        """Thông tin cơ bản"""
        print("\n" + "="*70)
        print("📊 BƯỚC 2: THÔNG TIN CƠ BẢN")
        print("="*70)
        
        print("\n1. KIỂU DỮ LIỆU:")
        print(self.df.dtypes)
        
        print("\n2. MISSING VALUES:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✓ Không có giá trị thiếu!")
        else:
            print(missing[missing > 0])
        
        print("\n3. DUPLICATE ROWS:")
        duplicates = self.df.duplicated().sum()
        print(f"{'✓' if duplicates == 0 else '⚠️'} {duplicates:,} dòng trùng lặp")
        
        print("\n4. UNIQUE VALUES:")
        for col in ['Airline', 'Origin', 'Destination', 'Class']:
            if col in self.df.columns:
                print(f"   {col}: {self.df[col].nunique()}")
        
        print("\n5. PHÂN PHỐI HẠNG GHẾ (Top 10):")
        class_counts = self.df['Class'].value_counts()
        total = len(self.df)
        for i, (cls, count) in enumerate(class_counts.head(10).items()):
            print(f"   {i+1}. {cls}: {count:,} ({count/total*100:.1f}%)")
    
    def statistical_summary(self):
        """Tóm tắt thống kê"""
        print("\n" + "="*70)
        print("📈 BƯỚC 3: TÓM TẮT THỐNG KÊ")
        print("="*70)
        
        print("\n1. MÔ TẢ GIÁ VÉ:")
        price_stats = self.df['Price_VND'].describe()
        print(f"   Mean:     {price_stats['mean']:,.0f} VND")
        print(f"   Median:   {price_stats['50%']:,.0f} VND")
        print(f"   Std:      {price_stats['std']:,.0f} VND")
        print(f"   Min:      {price_stats['min']:,.0f} VND")
        print(f"   Max:      {price_stats['max']:,.0f} VND")
        print(f"   Range:    {price_stats['max']-price_stats['min']:,.0f} VND")
        print(f"   IQR:      {price_stats['75%']-price_stats['25%']:,.0f} VND")
        
        print("\n2. SKEWNESS & KURTOSIS:")
        print(f"   Skewness: {self.df['Price_VND'].skew():.3f}")
        print(f"   Kurtosis: {self.df['Price_VND'].kurtosis():.3f}")
        
        if 'Duration_Minutes' in self.df.columns:
            print("\n3. THỜI GIAN BAY:")
            print(f"   Mean:     {self.df['Duration_Minutes'].mean():.0f} phút")
            print(f"   Median:   {self.df['Duration_Minutes'].median():.0f} phút")
            print(f"   Min:      {self.df['Duration_Minutes'].min():.0f} phút")
            print(f"   Max:      {self.df['Duration_Minutes'].max():.0f} phút")
    
    def plot_1_price_distribution(self):
        """Biểu đồ 1: Phân phối giá vé"""
        print("\n📊 Plot 1: Phân phối giá vé...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Histogram
        axes[0, 0].hist(self.df['Price_VND'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.df['Price_VND'].mean(), color='red', 
                          linestyle='--', linewidth=2, label=f'Mean: {self.df["Price_VND"].mean():,.0f}')
        axes[0, 0].axvline(self.df['Price_VND'].median(), color='green', 
                          linestyle='--', linewidth=2, label=f'Median: {self.df["Price_VND"].median():,.0f}')
        axes[0, 0].set_title('Histogram - Phan Phoi Gia Ve', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Gia Ve (VND)', fontsize=12)
        axes[0, 0].set_ylabel('Tan Suat', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(self.df['Price_VND'], vert=True)
        axes[0, 1].set_title('Box Plot - Phat Hien Outliers', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Gia Ve (VND)', fontsize=12)
        axes[0, 1].grid(alpha=0.3)
        
        # KDE plot
        self.df['Price_VND'].plot(kind='kde', ax=axes[1, 0], linewidth=2)
        axes[1, 0].set_title('KDE Plot - Mat Do Phan Phoi', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Gia Ve (VND)', fontsize=12)
        axes[1, 0].set_ylabel('Mat Do', fontsize=12)
        axes[1, 0].grid(alpha=0.3)
        
        # Q-Q plot
        stats.probplot(self.df['Price_VND'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot - Kiem Tra Chuan', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Đã lưu: 01_price_distribution.png")
    
    def plot_2_airline_analysis(self):
        """Biểu đồ 2: Phân tích theo hãng bay"""
        print("\n📊 Plot 2: Phân tích hãng bay...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Bar chart - Count
        airline_counts = self.df['Airline'].value_counts()
        airline_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue', edgecolor='black')
        axes[0, 0].set_title('So Luong Chuyen Bay Theo Hang', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Hang Bay', fontsize=12)
        axes[0, 0].set_ylabel('So Chuyen', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # Box plot - Price by airline
        self.df.boxplot(column='Price_VND', by='Airline', ax=axes[0, 1])
        axes[0, 1].set_title('Phan Phoi Gia Theo Hang Bay', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Hang Bay', fontsize=12)
        axes[0, 1].set_ylabel('Gia Ve (VND)', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        plt.suptitle('')
        
        # Bar chart - Average price
        airline_avg_price = self.df.groupby('Airline')['Price_VND'].mean().sort_values(ascending=False)
        airline_avg_price.plot(kind='bar', ax=axes[1, 0], color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('Gia Ve Trung Binh Theo Hang', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Hang Bay', fontsize=12)
        axes[1, 0].set_ylabel('Gia TB (VND)', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # Pie chart - Market share (chỉ Top 5, còn lại gộp vào "Others")
        top_n = 5  # Chỉ lấy Top 5 hãng bay
        
        # Lấy Top 5 hãng
        top_airlines = airline_counts.head(top_n)
        others = airline_counts.iloc[top_n:].sum()
        
        # Tạo series mới với Top 5 + Others
        if others > 0:
            airline_display = top_airlines.copy()
            airline_display['Others'] = others
        else:
            airline_display = top_airlines
        
        # Vẽ pie chart với màu đẹp
        colors = plt.cm.Set3(range(len(airline_display)))
        airline_display.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%', 
                            startangle=90, colors=colors)
        axes[1, 1].set_title('Thi Phan Cac Hang Bay', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_airline_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Đã lưu: 02_airline_analysis.png")
    
    def plot_3_route_analysis(self):
        """Biểu đồ 3: Phân tích tuyến bay"""
        print("\n📊 Plot 3: Phân tích tuyến bay...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top 10 routes by frequency
        routes = self.df['Origin'] + ' -> ' + self.df['Destination']
        top_routes = routes.value_counts().head(10)
        top_routes.plot(kind='barh', ax=axes[0, 0], color='lightgreen', edgecolor='black')
        axes[0, 0].set_title('Top 10 Tuyen Bay Pho Bien', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('So Chuyen', fontsize=12)
        axes[0, 0].set_ylabel('Tuyen Bay', fontsize=12)
        axes[0, 0].grid(alpha=0.3, axis='x')
        
        # Top 10 most expensive routes
        route_avg_price = self.df.groupby(['Origin', 'Destination'])['Price_VND'].mean()
        route_labels = route_avg_price.index.map(lambda x: f'{x[0]} -> {x[1]}')
        top_expensive = pd.Series(route_avg_price.values, index=route_labels).nlargest(10)
        top_expensive.plot(kind='barh', ax=axes[0, 1], color='salmon', edgecolor='black')
        axes[0, 1].set_title('Top 10 Tuyen Bay Dat Nhat', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Gia TB (VND)', fontsize=12)
        axes[0, 1].set_ylabel('Tuyen Bay', fontsize=12)
        axes[0, 1].grid(alpha=0.3, axis='x')
        
        # Origin distribution
        origin_counts = self.df['Origin'].value_counts().head(10)
        origin_counts.plot(kind='bar', ax=axes[1, 0], color='gold', edgecolor='black')
        axes[1, 0].set_title('Top 10 Diem Khoi Hanh', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('San Bay', fontsize=12)
        axes[1, 0].set_ylabel('So Chuyen', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # Destination distribution
        dest_counts = self.df['Destination'].value_counts().head(10)
        dest_counts.plot(kind='bar', ax=axes[1, 1], color='orchid', edgecolor='black')
        axes[1, 1].set_title('Top 10 Diem Den', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('San Bay', fontsize=12)
        axes[1, 1].set_ylabel('So Chuyen', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_route_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Đã lưu: 03_route_analysis.png")
    
    def plot_4_time_analysis(self):
        """Biểu đồ 4: Phân tích thời gian"""
        print("\n📊 Plot 4: Phân tích thời gian...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Price by month
        if 'Month' in self.df.columns:
            monthly_price = self.df.groupby('Month')['Price_VND'].mean()
            monthly_price.plot(kind='line', ax=axes[0, 0], marker='o', linewidth=2, markersize=8)
            axes[0, 0].set_title('Gia TB Theo Thang', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Thang', fontsize=12)
            axes[0, 0].set_ylabel('Gia TB (VND)', fontsize=12)
            axes[0, 0].grid(alpha=0.3)
        
        # Price by weekday
        if 'Weekday' in self.df.columns:
            weekday_price = self.df.groupby('Weekday')['Price_VND'].mean()
            weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekday_price.plot(kind='bar', ax=axes[0, 1], color='teal', edgecolor='black')
            axes[0, 1].set_title('Gia TB Theo Ngay Trong Tuan', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Ngay', fontsize=12)
            axes[0, 1].set_ylabel('Gia TB (VND)', fontsize=12)
            axes[0, 1].set_xticklabels(weekday_labels[:len(weekday_price)], rotation=0)
            axes[0, 1].grid(alpha=0.3, axis='y')
        
        # Price by departure hour
        if 'Departure_Hour' in self.df.columns:
            hourly_price = self.df.groupby('Departure_Hour')['Price_VND'].mean()
            hourly_price.plot(kind='line', ax=axes[0, 2], marker='s', linewidth=2, markersize=6)
            axes[0, 2].set_title('Gia TB Theo Gio Khoi Hanh', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Gio', fontsize=12)
            axes[0, 2].set_ylabel('Gia TB (VND)', fontsize=12)
            axes[0, 2].grid(alpha=0.3)
        
        # Flight count by month
        if 'Month' in self.df.columns:
            monthly_count = self.df['Month'].value_counts().sort_index()
            monthly_count.plot(kind='bar', ax=axes[1, 0], color='steelblue', edgecolor='black')
            axes[1, 0].set_title('So Chuyen Bay Theo Thang', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Thang', fontsize=12)
            axes[1, 0].set_ylabel('So Chuyen', fontsize=12)
            axes[1, 0].tick_params(axis='x', rotation=0)
            axes[1, 0].grid(alpha=0.3, axis='y')
        
        # Duration distribution
        if 'Duration_Minutes' in self.df.columns:
            axes[1, 1].hist(self.df['Duration_Minutes'], bins=30, edgecolor='black', alpha=0.7)
            axes[1, 1].axvline(self.df['Duration_Minutes'].mean(), color='red', 
                              linestyle='--', linewidth=2, label=f'Mean: {self.df["Duration_Minutes"].mean():.0f} min')
            axes[1, 1].set_title('Phan Phoi Thoi Gian Bay', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Thoi Gian (phut)', fontsize=12)
            axes[1, 1].set_ylabel('Tan Suat', fontsize=12)
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        # Price vs Duration scatter
        if 'Duration_Minutes' in self.df.columns:
            axes[1, 2].scatter(self.df['Duration_Minutes'], self.df['Price_VND'], alpha=0.5, s=20)
            axes[1, 2].set_title('Gia Ve vs Thoi Gian Bay', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Thoi Gian Bay (phut)', fontsize=12)
            axes[1, 2].set_ylabel('Gia Ve (VND)', fontsize=12)
            axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Đã lưu: 04_time_analysis.png")
    
    def plot_5_class_comparison(self):
        """Biểu đồ 5: So sánh hạng ghế"""
        print("\n📊 Plot 5: So sánh hạng ghế...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get top classes for better visualization
        top_classes = self.df['Class'].value_counts().head(8).index.tolist()
        df_top = self.df[self.df['Class'].isin(top_classes)]
        
        # Box plot comparison
        df_top.boxplot(column='Price_VND', by='Class', ax=axes[0, 0], rot=45)
        axes[0, 0].set_title('So Sanh Gia Theo Hang Ghe (Top 8)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Hang Ghe', fontsize=12)
        axes[0, 0].set_ylabel('Gia Ve (VND)', fontsize=12)
        plt.suptitle('')
        
        # Bar chart - Average price (all classes)
        class_avg = self.df.groupby('Class')['Price_VND'].mean().sort_values(ascending=False).head(10)
        class_avg.plot(kind='barh', ax=axes[0, 1], color='salmon', edgecolor='black')
        axes[0, 1].set_title('Top 10 Hang Ghe Dat Nhat (Gia TB)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Gia TB (VND)', fontsize=12)
        axes[0, 1].set_ylabel('Hang Ghe', fontsize=12)
        axes[0, 1].grid(alpha=0.3, axis='x')
        
        # Distribution comparison (top 5 classes only)
        top5_classes = self.df['Class'].value_counts().head(5).index.tolist()
        for cls in top5_classes:
            data = self.df[self.df['Class'] == cls]['Price_VND']
            axes[1, 0].hist(data, bins=30, alpha=0.5, label=cls[:20], edgecolor='black')
        axes[1, 0].set_title('Phan Phoi Gia Theo Hang Ghe (Top 5)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Gia Ve (VND)', fontsize=12)
        axes[1, 0].set_ylabel('Tan Suat', fontsize=12)
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(alpha=0.3)
        
        # Pie chart - Class distribution (gộp các hạng < 2% vào "Khác")
        class_dist = self.df['Class'].value_counts()
        total_tickets = class_dist.sum()
        threshold = total_tickets * 0.02  # 2% threshold cho hạng ghế
        
        # Tách các hạng lớn và nhỏ
        major_classes = class_dist[class_dist >= threshold]
        minor_classes = class_dist[class_dist < threshold]
        
        # Nếu có hạng nhỏ, gộp lại
        if len(minor_classes) > 0:
            class_display = major_classes.copy()
            class_display['Khác'] = minor_classes.sum()
        else:
            class_display = major_classes
        
        # Vẽ pie chart với labels ngắn gọn
        labels_short = [cls[:20] + '...' if len(cls) > 20 else cls for cls in class_display.index]
        colors = plt.cm.Set3(range(len(class_display)))
        axes[1, 1].pie(class_display, labels=labels_short, 
                      autopct='%1.1f%%', startangle=90, colors=colors)
        axes[1, 1].set_title('Phan Bo Hang Ghe', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_class_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Đã lưu: 05_class_comparison.png")
    
    def plot_6_correlation_analysis(self):
        """Biểu đồ 6: Phân tích tương quan"""
        print("\n📊 Plot 6: Phân tích tương quan...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Correlation heatmap
        numeric_cols = []
        for col in ['Day', 'Month', 'Weekday', 'Departure_Hour', 
                   'Arrival_Hour', 'Duration_Minutes', 'Stops', 'Baggage_kg', 'Price_VND']:
            if col in self.df.columns:
                numeric_cols.append(col)
        
        corr = self.df[numeric_cols].corr()
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title('Ma Tran Tuong Quan', fontsize=14, fontweight='bold')
        
        # Correlation with price
        price_corr = corr['Price_VND'].drop('Price_VND').sort_values(ascending=False)
        price_corr.plot(kind='barh', ax=axes[1], color='steelblue', edgecolor='black')
        axes[1].set_title('Tuong Quan Voi Gia Ve', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('He So Tuong Quan', fontsize=12)
        axes[1].set_ylabel('Features', fontsize=12)
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
        axes[1].grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Đã lưu: 06_correlation_analysis.png")
    
    def plot_7_outlier_analysis(self):
        """Biểu đồ 7: Phân tích outliers"""
        print("\n📊 Plot 7: Phân tích outliers...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Z-score
        z_scores = np.abs(stats.zscore(self.df['Price_VND']))
        outliers_z = (z_scores > 3).sum()
        
        axes[0, 0].scatter(range(len(z_scores)), z_scores, alpha=0.5, s=20)
        axes[0, 0].axhline(y=3, color='red', linestyle='--', linewidth=2, label='Threshold (Z=3)')
        axes[0, 0].set_title(f'Z-Score Analysis ({outliers_z} outliers)', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Index', fontsize=12)
        axes[0, 0].set_ylabel('Z-Score', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # IQR method
        Q1 = self.df['Price_VND'].quantile(0.25)
        Q3 = self.df['Price_VND'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers_iqr = ((self.df['Price_VND'] < lower) | (self.df['Price_VND'] > upper)).sum()
        
        axes[0, 1].boxplot(self.df['Price_VND'], vert=True)
        axes[0, 1].set_title(f'Box Plot - IQR Method ({outliers_iqr} outliers)', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Gia Ve (VND)', fontsize=12)
        axes[0, 1].grid(alpha=0.3)
        
        # Histogram with outliers highlighted
        axes[1, 0].hist(self.df['Price_VND'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(lower, color='red', linestyle='--', linewidth=2, label=f'Lower: {lower:,.0f}')
        axes[1, 0].axvline(upper, color='red', linestyle='--', linewidth=2, label=f'Upper: {upper:,.0f}')
        axes[1, 0].set_title('Phan Phoi Voi Nguong IQR', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Gia Ve (VND)', fontsize=12)
        axes[1, 0].set_ylabel('Tan Suat', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Percentile plot
        percentiles = np.percentile(self.df['Price_VND'], range(0, 101, 5))
        axes[1, 1].plot(range(0, 101, 5), percentiles, marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_title('Percentile Plot', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Percentile', fontsize=12)
        axes[1, 1].set_ylabel('Gia Ve (VND)', fontsize=12)
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Đã lưu: 07_outlier_analysis.png")
    
    def generate_summary_report(self):
        """Tạo báo cáo tổng kết"""
        print("\n" + "="*70)
        print("📝 BƯỚC 4: TẠO BÁO CÁO TỔNG KẾT")
        print("="*70)
        
        # Tính Q1, Q3, IQR cho phần outliers
        Q1 = self.df['Price_VND'].quantile(0.25)
        Q3 = self.df['Price_VND'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Tính số hạng ghế
        num_classes = self.df['Class'].nunique()
        top_classes = self.df['Class'].value_counts().head(5)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           BÁO CÁO PHÂN TÍCH DỮ LIỆU VÉ MÁY BAY (EDA)             ║
╚══════════════════════════════════════════════════════════════════╝

1. TỔNG QUAN DỮ LIỆU
   • Tổng số mẫu:         {len(self.df):,} dòng
   • Số features:         {len(self.df.columns)} cột
   • Kích thước:          {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   • Missing values:      {self.df.isnull().sum().sum()}
   • Duplicate rows:      {self.df.duplicated().sum()}

2. THỐNG KÊ GIÁ VÉ
   • Giá TB:              {self.df['Price_VND'].mean():,.0f} VND
   • Giá median:          {self.df['Price_VND'].median():,.0f} VND
   • Độ lệch chuẩn:       {self.df['Price_VND'].std():,.0f} VND
   • Giá min:             {self.df['Price_VND'].min():,.0f} VND
   • Giá max:             {self.df['Price_VND'].max():,.0f} VND
   • Skewness:            {self.df['Price_VND'].skew():.3f}
   • Kurtosis:            {self.df['Price_VND'].kurtosis():.3f}

3. HÃNG HÀNG KHÔNG
   • Số hãng:             {self.df['Airline'].nunique()}
   • Hãng lớn nhất:       {self.df['Airline'].value_counts().index[0]}
   • Số chuyến:           {self.df['Airline'].value_counts().iloc[0]:,}

4. TUYẾN BAY
   • Số điểm khởi hành:   {self.df['Origin'].nunique()}
   • Số điểm đến:         {self.df['Destination'].nunique()}
   • Tuyến phổ biến:      {(self.df['Origin'] + '-' + self.df['Destination']).value_counts().index[0]}

5. THỜI GIAN
   • Thời gian bay TB:    {self.df['Duration_Minutes'].mean():.0f} phút
   • Giờ khởi hành TB:    {self.df['Departure_Hour'].mean():.1f}
   • Tháng có nhiều bay:  {self.df['Month'].value_counts().index[0]}

6. HẠNG VÉ
   • Tổng số loại hạng:   {num_classes}
   • Top 5 hạng phổ biến:
{chr(10).join([f'     - {cls}: {count:,} ({count/len(self.df)*100:.1f}%)' for cls, count in top_classes.items()])}

7. OUTLIERS
   • Z-score (>3):        {(np.abs(stats.zscore(self.df['Price_VND'])) > 3).sum()}
   • IQR method:          {((self.df['Price_VND'] < Q1-1.5*IQR) | (self.df['Price_VND'] > Q3+1.5*IQR)).sum()}

8. TOP 5 HÃNG BAY (THEO SỐ CHUYẾN)
{chr(10).join([f'   {i+1}. {airline}: {count:,} chuyến' for i, (airline, count) in enumerate(self.df['Airline'].value_counts().head().items())])}

9. TOP 5 TUYẾN BAY PHỔ BIẾN
{chr(10).join([f'   {i+1}. {route}: {count:,} chuyến' for i, (route, count) in enumerate((self.df['Origin'] + ' → ' + self.df['Destination']).value_counts().head().items())])}

10. BIỂU ĐỒ ĐÃ TẠO
   ✓ 01_price_distribution.png    - Phân phối giá vé
   ✓ 02_airline_analysis.png       - Phân tích hãng bay
   ✓ 03_route_analysis.png         - Phân tích tuyến bay
   ✓ 04_time_analysis.png          - Phân tích thời gian
   ✓ 05_class_comparison.png       - So sánh hạng ghế
   ✓ 06_correlation_analysis.png   - Phân tích tương quan
   ✓ 07_outlier_analysis.png       - Phân tích outliers

╔══════════════════════════════════════════════════════════════════╗
║                     KẾT THÚC BÁO CÁO EDA                         ║
╚══════════════════════════════════════════════════════════════════╝
        """
        
        print(report)
        
        # Lưu báo cáo
        with open(f'{self.output_dir}/00_EDA_REPORT.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Đã lưu báo cáo: {self.output_dir}/00_EDA_REPORT.txt")
    
    def run_full_eda(self):
        """Chạy toàn bộ EDA"""
        print("\n" + "="*70)
        print("🚀 KHỞI ĐỘNG EDA - EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        self.load_data()
        self.basic_info()
        self.statistical_summary()
        
        print("\n" + "="*70)
        print("📊 TẠO BIỂU ĐỒ (7 files)")
        print("="*70)
        
        self.plot_1_price_distribution()
        self.plot_2_airline_analysis()
        self.plot_3_route_analysis()
        self.plot_4_time_analysis()
        self.plot_5_class_comparison()
        self.plot_6_correlation_analysis()
        self.plot_7_outlier_analysis()
        
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("🎉 HOÀN THÀNH EDA!")
        print("="*70)
        import os
        plot_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        print(f"✅ Đã tạo {len(plot_files)} biểu đồ")
        print(f"✅ Đã lưu tất cả vào thư mục: {self.output_dir}/")
        print("="*70)

def main():
    """Hàm chính"""
    import os
    import sys
    
    # Chọn file để phân tích
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Mặc định: dùng augmented data nếu có
        if os.path.exists('Flight_Price_Data_Enhanced_Up.csv'):
            data_file = 'Flight_Price_Data_Enhanced_Up.csv'
            print("\n✓ Phân tích dữ liệu Flight_Price_Data_Enhanced_Up.csv")
        elif os.path.exists('/mnt/user-data/uploads/Flight_Price_Data_Enhanced_Up.csv'):
            data_file = '/mnt/user-data/uploads/Flight_Price_Data_Enhanced_Up.csv'
            print("\n✓ Phân tích dữ liệu từ uploads")
        else:
            print("\n❌ Không tìm thấy file dữ liệu!")
            print("Cần có: Flight_Price_Data_Enhanced_Up.csv")
            return
    
    print(f"📂 File đầu vào: {data_file}")
    
    # Khởi tạo và chạy EDA
    eda = FlightDataEDA(data_file, output_dir='eda_plots')
    eda.run_full_eda()
    
    print("\n📌 Xem các biểu đồ trong thư mục: eda_plots/")
    print("📌 Đọc báo cáo tổng kết: eda_plots/00_EDA_REPORT.txt")

if __name__ == "__main__":
    main()