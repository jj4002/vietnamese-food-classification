import cv2
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import json
from datetime import datetime
import base64
from io import BytesIO

# Kiểm tra GPU support
try:
    from cuml.decomposition import PCA as cumlPCA  # type: ignore
    CUML_AVAILABLE = True
    print("GPU detected - có thể sử dụng cuML PCA")
except ImportError:
    CUML_AVAILABLE = False
    print("Using CPU - sklearn PCA")

# Đường dẫn đến ảnh
image_path = "dataset/Train/Banh chung/1.jpg"

# Đọc ảnh
print(f"Đang đọc ảnh: {image_path}")
img = cv2.imread(image_path)

if img is None:
    print(f"Không thể đọc ảnh từ: {image_path}")
    exit(1)

print(f"Kích thước ảnh gốc: {img.shape}")

# Lưu thông tin ảnh gốc để so sánh
img_original = img.copy()
original_height, original_width = img.shape[:2]
original_size = original_height * original_width
original_aspect_ratio = original_width / original_height if original_height > 0 else 0

# Resize về 256x256
img_resized = cv2.resize(img, (256, 256))
resized_height, resized_width = img_resized.shape[:2]
resized_size = resized_height * resized_width
resized_aspect_ratio = resized_width / resized_height if resized_height > 0 else 0

print(f"Kích thước sau resize: {img_resized.shape}\n")

# ========== SO SÁNH ẢNH TRƯỚC VÀ SAU XỬ LÝ ==========
print("="*60)
print("SO SÁNH ẢNH TRƯỚC VÀ SAU XỬ LÝ")
print("="*60)

print("\n1. THÔNG TIN KÍCH THƯỚC:")
print("-" * 60)
print(f"  {'Thông số':<25} {'Trước xử lý':<20} {'Sau resize (256x256)':<20} {'Thay đổi':<15}")
print(f"  {'-'*25} {'-'*20} {'-'*20} {'-'*15}")
print(f"  {'Chiều rộng (width)':<25} {original_width:<20} {resized_width:<20} {resized_width - original_width:+.0f}")
print(f"  {'Chiều cao (height)':<25} {original_height:<20} {resized_height:<20} {resized_height - original_height:+.0f}")
print(f"  {'Tổng số pixel':<25} {original_size:<20} {resized_size:<20} {resized_size - original_size:+.0f} ({(resized_size/original_size - 1)*100:+.2f}%)")
print(f"  {'Aspect ratio':<25} {original_aspect_ratio:.4f}{'':<15} {resized_aspect_ratio:.4f}{'':<15} {resized_aspect_ratio - original_aspect_ratio:+.4f}")

print("\n2. THỐNG KÊ PIXEL (theo kênh màu):")
print("-" * 60)
print(f"  {'Kênh':<10} {'Thông số':<15} {'Trước xử lý':<20} {'Sau resize':<20} {'Thay đổi':<15}")
print(f"  {'-'*10} {'-'*15} {'-'*20} {'-'*20} {'-'*15}")

for i, channel_name in enumerate(['B (Blue)', 'G (Green)', 'R (Red)']):
    orig_channel = img_original[:, :, i]
    resized_channel = img_resized[:, :, i]
    
    orig_mean = np.mean(orig_channel)
    resized_mean = np.mean(resized_channel)
    
    orig_std = np.std(orig_channel)
    resized_std = np.std(resized_channel)
    
    orig_var = np.var(orig_channel)
    resized_var = np.var(resized_channel)
    
    orig_min = np.min(orig_channel)
    resized_min = np.min(resized_channel)
    
    orig_max = np.max(orig_channel)
    resized_max = np.max(resized_channel)
    
    print(f"  {channel_name:<10} {'Mean':<15} {orig_mean:>8.2f}{'':<11} {resized_mean:>8.2f}{'':<11} {resized_mean - orig_mean:>+8.2f}")
    print(f"  {'':<10} {'Std':<15} {orig_std:>8.2f}{'':<11} {resized_std:>8.2f}{'':<11} {resized_std - orig_std:>+8.2f}")
    print(f"  {'':<10} {'Variance':<15} {orig_var:>8.2f}{'':<11} {resized_var:>8.2f}{'':<11} {resized_var - orig_var:>+8.2f}")
    print(f"  {'':<10} {'Min':<15} {orig_min:>8.0f}{'':<11} {resized_min:>8.0f}{'':<11} {resized_min - orig_min:>+8.0f}")
    print(f"  {'':<10} {'Max':<15} {orig_max:>8.0f}{'':<11} {resized_max:>8.0f}{'':<11} {resized_max - orig_max:>+8.0f}")
    if i < 2:
        print()

print("\n3. THỐNG KÊ TỔNG THỂ:")
print("-" * 60)
orig_total_mean = np.mean(img_original)
resized_total_mean = np.mean(img_resized)
orig_total_std = np.std(img_original)
resized_total_std = np.std(img_resized)
orig_total_var = np.var(img_original)
resized_total_var = np.var(img_resized)
orig_total_min = np.min(img_original)
resized_total_min = np.min(img_resized)
orig_total_max = np.max(img_original)
resized_total_max = np.max(img_resized)

print(f"  {'Thông số':<15} {'Trước xử lý':<20} {'Sau resize':<20} {'Thay đổi':<15}")
print(f"  {'-'*15} {'-'*20} {'-'*20} {'-'*15}")
print(f"  {'Mean':<15} {orig_total_mean:>8.2f}{'':<11} {resized_total_mean:>8.2f}{'':<11} {resized_total_mean - orig_total_mean:>+8.2f}")
print(f"  {'Std':<15} {orig_total_std:>8.2f}{'':<11} {resized_total_std:>8.2f}{'':<11} {resized_total_std - orig_total_std:>+8.2f}")
print(f"  {'Variance':<15} {orig_total_var:>8.2f}{'':<11} {resized_total_var:>8.2f}{'':<11} {resized_total_var - orig_total_var:>+8.2f}")
print(f"  {'Min':<15} {orig_total_min:>8.0f}{'':<11} {resized_total_min:>8.0f}{'':<11} {resized_total_min - orig_total_min:>+8.0f}")
print(f"  {'Max':<15} {orig_total_max:>8.0f}{'':<11} {resized_total_max:>8.0f}{'':<11} {resized_total_max - orig_total_max:>+8.0f}")

print("\n4. TỶ LỆ THAY ĐỔI:")
print("-" * 60)
size_ratio = resized_size / original_size
mean_ratio = resized_total_mean / orig_total_mean if orig_total_mean > 0 else 0
std_ratio = resized_total_std / orig_total_std if orig_total_std > 0 else 0
var_ratio = resized_total_var / orig_total_var if orig_total_var > 0 else 0

print(f"  Số pixel:        {size_ratio:.4f}x ({size_ratio*100:.2f}%)")
print(f"  Mean:            {mean_ratio:.4f}x ({mean_ratio*100:.2f}%)")
print(f"  Std:             {std_ratio:.4f}x ({std_ratio*100:.2f}%)")
print(f"  Variance:        {var_ratio:.4f}x ({var_ratio*100:.2f}%)")

print("\n" + "="*60)
print()

# ========== VẼ BIỂU ĐỒ SO SÁNH BẰNG MATPLOTLIB ==========
print("\n" + "="*60)
print("VẼ BIỂU ĐỒ SO SÁNH")
print("="*60)

try:
    # 1. Biểu đồ so sánh ảnh
    print("\n1. Vẽ biểu đồ so sánh ảnh...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Chuyển BGR sang RGB để hiển thị đúng
    img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(img_original_rgb)
    axes[0].set_title(f'Ảnh gốc\n{original_width}x{original_height} pixels\n{original_size:,} pixels', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_resized_rgb)
    axes[1].set_title(f'Ảnh sau resize\n{resized_width}x{resized_height} pixels\n{resized_size:,} pixels', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('So sánh ảnh trước và sau xử lý', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('comparison_images.png', dpi=150, bbox_inches='tight')
    print("  ✓ Đã lưu: comparison_images.png")
    plt.close()
    
    # 2. Biểu đồ so sánh thống kê theo kênh màu
    print("2. Vẽ biểu đồ so sánh thống kê theo kênh màu...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    channels = ['Blue', 'Green', 'Red']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Tính toán thống kê
    orig_stats = {'mean': [], 'std': [], 'variance': []}
    resized_stats = {'mean': [], 'std': [], 'variance': []}
    
    for i, (ch_name, color) in enumerate(zip(channels, colors)):
        orig_ch = img_original[:, :, i]
        resized_ch = img_resized[:, :, i]
        
        orig_stats['mean'].append(np.mean(orig_ch))
        orig_stats['std'].append(np.std(orig_ch))
        orig_stats['variance'].append(np.var(orig_ch))
        
        resized_stats['mean'].append(np.mean(resized_ch))
        resized_stats['std'].append(np.std(resized_ch))
        resized_stats['variance'].append(np.var(resized_ch))
    
    # Mean
    x = np.arange(len(channels))
    width = 0.35
    axes[0, 0].bar(x - width/2, orig_stats['mean'], width, label='Trước xử lý', color='#3498db', alpha=0.8)
    axes[0, 0].bar(x + width/2, resized_stats['mean'], width, label='Sau resize', color='#e74c3c', alpha=0.8)
    axes[0, 0].set_xlabel('Kênh màu', fontsize=11)
    axes[0, 0].set_ylabel('Mean', fontsize=11)
    axes[0, 0].set_title('So sánh Mean theo kênh màu', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(channels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std
    axes[0, 1].bar(x - width/2, orig_stats['std'], width, label='Trước xử lý', color='#3498db', alpha=0.8)
    axes[0, 1].bar(x + width/2, resized_stats['std'], width, label='Sau resize', color='#e74c3c', alpha=0.8)
    axes[0, 1].set_xlabel('Kênh màu', fontsize=11)
    axes[0, 1].set_ylabel('Standard Deviation', fontsize=11)
    axes[0, 1].set_title('So sánh Std theo kênh màu', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(channels)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Variance
    axes[1, 0].bar(x - width/2, orig_stats['variance'], width, label='Trước xử lý', color='#3498db', alpha=0.8)
    axes[1, 0].bar(x + width/2, resized_stats['variance'], width, label='Sau resize', color='#e74c3c', alpha=0.8)
    axes[1, 0].set_xlabel('Kênh màu', fontsize=11)
    axes[1, 0].set_ylabel('Variance', fontsize=11)
    axes[1, 0].set_title('So sánh Variance theo kênh màu', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(channels)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Tỷ lệ thay đổi
    change_ratios = {
        'Mean': [resized_stats['mean'][i]/orig_stats['mean'][i] if orig_stats['mean'][i] > 0 else 0 
                 for i in range(3)],
        'Std': [resized_stats['std'][i]/orig_stats['std'][i] if orig_stats['std'][i] > 0 else 0 
                for i in range(3)],
        'Variance': [resized_stats['variance'][i]/orig_stats['variance'][i] if orig_stats['variance'][i] > 0 else 0 
                     for i in range(3)]
    }
    
    x2 = np.arange(len(channels))
    width2 = 0.25
    axes[1, 1].bar(x2 - width2, change_ratios['Mean'], width2, label='Mean', color='#3498db', alpha=0.8)
    axes[1, 1].bar(x2, change_ratios['Std'], width2, label='Std', color='#2ecc71', alpha=0.8)
    axes[1, 1].bar(x2 + width2, change_ratios['Variance'], width2, label='Variance', color='#e74c3c', alpha=0.8)
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Kênh màu', fontsize=11)
    axes[1, 1].set_ylabel('Tỷ lệ (Sau/Trước)', fontsize=11)
    axes[1, 1].set_title('Tỷ lệ thay đổi theo kênh màu', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels(channels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('So sánh thống kê theo kênh màu', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('comparison_statistics.png', dpi=150, bbox_inches='tight')
    print("  ✓ Đã lưu: comparison_statistics.png")
    plt.close()
    
    # 3. Biểu đồ histogram so sánh phân bố pixel
    print("3. Vẽ biểu đồ histogram so sánh phân bố pixel...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for i, (ch_name, color) in enumerate(zip(channels, colors)):
        orig_ch = img_original[:, :, i].flatten()
        resized_ch = img_resized[:, :, i].flatten()
        
        # Histogram cho ảnh gốc
        axes[0, i].hist(orig_ch, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0, i].set_title(f'Ảnh gốc - Kênh {ch_name}\nMean: {np.mean(orig_ch):.2f}, Std: {np.std(orig_ch):.2f}', 
                            fontsize=11, fontweight='bold')
        axes[0, i].set_xlabel('Giá trị pixel', fontsize=10)
        axes[0, i].set_ylabel('Tần suất', fontsize=10)
        axes[0, i].grid(True, alpha=0.3)
        
        # Histogram cho ảnh sau resize
        axes[1, i].hist(resized_ch, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[1, i].set_title(f'Ảnh sau resize - Kênh {ch_name}\nMean: {np.mean(resized_ch):.2f}, Std: {np.std(resized_ch):.2f}', 
                            fontsize=11, fontweight='bold')
        axes[1, i].set_xlabel('Giá trị pixel', fontsize=10)
        axes[1, i].set_ylabel('Tần suất', fontsize=10)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle('So sánh phân bố pixel theo kênh màu', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('comparison_histogram.png', dpi=150, bbox_inches='tight')
    print("  ✓ Đã lưu: comparison_histogram.png")
    plt.close()
    
    # 4. Biểu đồ so sánh kích thước và tổng quan
    print("4. Vẽ biểu đồ so sánh kích thước và tổng quan...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # So sánh kích thước
    categories = ['Width', 'Height', 'Total Pixels']
    orig_values = [original_width, original_height, original_size/1000]  # Chia 1000 để dễ nhìn
    resized_values = [resized_width, resized_height, resized_size/1000]
    
    x3 = np.arange(len(categories))
    axes[0, 0].bar(x3 - width/2, orig_values, width, label='Trước xử lý', color='#3498db', alpha=0.8)
    axes[0, 0].bar(x3 + width/2, resized_values, width, label='Sau resize', color='#e74c3c', alpha=0.8)
    axes[0, 0].set_ylabel('Giá trị', fontsize=11)
    axes[0, 0].set_title('So sánh kích thước', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x3)
    axes[0, 0].set_xticklabels(categories)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel('Width/Height (px) hoặc Pixels (×1000)', fontsize=10)
    
    # So sánh thống kê tổng thể
    stats_categories = ['Mean', 'Std', 'Variance']
    orig_total_stats = [orig_total_mean, orig_total_std, orig_total_var/100]  # Chia variance để dễ nhìn
    resized_total_stats = [resized_total_mean, resized_total_std, resized_total_var/100]
    
    x4 = np.arange(len(stats_categories))
    axes[0, 1].bar(x4 - width/2, orig_total_stats, width, label='Trước xử lý', color='#3498db', alpha=0.8)
    axes[0, 1].bar(x4 + width/2, resized_total_stats, width, label='Sau resize', color='#e74c3c', alpha=0.8)
    axes[0, 1].set_ylabel('Giá trị', fontsize=11)
    axes[0, 1].set_title('So sánh thống kê tổng thể', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x4)
    axes[0, 1].set_xticklabels(stats_categories)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylabel('Mean/Std hoặc Variance (×100)', fontsize=10)
    
    # Pie chart so sánh số pixel
    sizes = [original_size, resized_size]
    labels = [f'Trước xử lý\n{original_size:,} pixels\n({original_width}×{original_height})',
              f'Sau resize\n{resized_size:,} pixels\n({resized_width}×{resized_height})']
    colors_pie = ['#3498db', '#e74c3c']
    axes[1, 0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
                   startangle=90, textprops={'fontsize': 10})
    axes[1, 0].set_title('So sánh số pixel', fontsize=12, fontweight='bold')
    
    # Tỷ lệ thay đổi tổng thể
    change_total = {
        'Mean': mean_ratio,
        'Std': std_ratio,
        'Variance': var_ratio,
        'Size': size_ratio
    }
    
    x5 = np.arange(len(change_total))
    axes[1, 1].bar(x5, list(change_total.values()), color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8)
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Không thay đổi')
    axes[1, 1].set_ylabel('Tỷ lệ (Sau/Trước)', fontsize=11)
    axes[1, 1].set_title('Tỷ lệ thay đổi tổng thể', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x5)
    axes[1, 1].set_xticklabels(list(change_total.keys()))
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('So sánh tổng quan ảnh trước và sau xử lý', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('comparison_overview.png', dpi=150, bbox_inches='tight')
    print("  ✓ Đã lưu: comparison_overview.png")
    plt.close()
    
    print("\n✓ Đã tạo tất cả biểu đồ so sánh!")
    print("  - comparison_images.png: So sánh ảnh trực quan")
    print("  - comparison_statistics.png: So sánh thống kê theo kênh màu")
    print("  - comparison_histogram.png: So sánh phân bố pixel")
    print("  - comparison_overview.png: So sánh tổng quan")
    
except Exception as e:
    print(f"⚠ Lỗi khi vẽ biểu đồ: {e}")
    import traceback
    traceback.print_exc()

# Tính số feature (sử dụng logic tương tự như trong V10_Balanced_Final.py)
def extract_features_detailed(image):
    """Trích xuất features từ ảnh và trả về từng loại feature riêng biệt"""
    features_dict = {}
    
    # ========== 3.1. COLOR FEATURES ==========
    print("="*60)
    print("BƯỚC 3: FEATURE EXTRACTION")
    print("="*60)
    print("\n3.1. COLOR FEATURES (Đặc trưng màu sắc):")
    print("-" * 60)
    
    # RGB Histogram
    rgb_features = []
    for i, channel_name in enumerate(['B', 'G', 'R']):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        rgb_features.extend(hist)
    features_dict['rgb_histogram'] = np.array(rgb_features, dtype=np.float32)
    print(f"  ✓ RGB Histogram: {len(rgb_features)} features (32 bins × 3 kênh)")
    print(f"    - Mean: {np.mean(rgb_features):.6f}, Std: {np.std(rgb_features):.6f}, Variance: {np.var(rgb_features):.6f}")
    
    # HSV Histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_features = []
    for i, channel_name in enumerate(['H', 'S', 'V']):
        hist = cv2.calcHist([hsv], [i], None, [32], [0, 256] if i > 0 else [0, 180])
        hist = cv2.normalize(hist, hist).flatten()
        hsv_features.extend(hist)
    features_dict['hsv_histogram'] = np.array(hsv_features, dtype=np.float32)
    print(f"  ✓ HSV Histogram: {len(hsv_features)} features (32 bins × 3 kênh)")
    print(f"    - Mean: {np.mean(hsv_features):.6f}, Std: {np.std(hsv_features):.6f}, Variance: {np.var(hsv_features):.6f}")
    
    # LAB Histogram
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_features = []
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [24], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        lab_features.extend(hist)
    features_dict['lab_histogram'] = np.array(lab_features, dtype=np.float32)
    print(f"  ✓ LAB Histogram: {len(lab_features)} features (24 bins × 3 kênh)")
    print(f"    - Mean: {np.mean(lab_features):.6f}, Std: {np.std(lab_features):.6f}, Variance: {np.var(lab_features):.6f}")
    
    # Color Moments (Mean, Std, Skewness)
    color_moments = []
    channel_names = ['B', 'G', 'R']
    for i, ch_name in enumerate(channel_names):
        channel = image[:, :, i].flatten()
        mean_val = np.mean(channel) / 255.0
        std_val = np.std(channel) / 255.0
        skew_val = skew(channel) / 10.0
        color_moments.extend([mean_val, std_val, skew_val])
        print(f"    - Kênh {ch_name}: Mean={mean_val:.4f}, Std={std_val:.4f}, Skewness={skew_val:.4f}")
    features_dict['color_moments'] = np.array(color_moments, dtype=np.float32)
    print(f"  ✓ Color Moments: {len(color_moments)} features (Mean, Std, Skewness × 3 kênh)")
    print(f"    - Variance: {np.var(color_moments):.6f}")
    
    # Color Ratios
    b, g, r = cv2.split(image.astype(np.float32) + 1e-6)
    color_ratios = [
        np.mean(r / (b + g)),
        np.mean(g / (r + b)),
        np.mean(b / (r + g))
    ]
    features_dict['color_ratios'] = np.array(color_ratios, dtype=np.float32)
    print(f"  ✓ Color Ratios: {len(color_ratios)} features")
    print(f"    - R/(B+G)={color_ratios[0]:.4f}, G/(R+B)={color_ratios[1]:.4f}, B/(R+G)={color_ratios[2]:.4f}")
    print(f"    - Variance: {np.var(color_ratios):.6f}")
    
    # ========== 3.2. TEXTURE FEATURES ==========
    print("\n3.2. TEXTURE FEATURES (Đặc trưng kết cấu):")
    print("-" * 60)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # HOG features
    hog_img = cv2.resize(gray, (64, 64))
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    hog_features = hog.compute(hog_img).flatten()
    features_dict['hog'] = hog_features.astype(np.float32)
    print(f"  ✓ HOG (Histogram of Oriented Gradients): {len(hog_features)} features")
    print(f"    - Mean: {np.mean(hog_features):.6f}, Std: {np.std(hog_features):.6f}, Variance: {np.var(hog_features):.6f}")
    
    # LBP features
    def calculate_lbp_fast(img):
        h, w = img.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = img[i, j]
                code = 0
                code |= (img[i-1, j-1] >= center) << 7
                code |= (img[i-1, j  ] >= center) << 6
                code |= (img[i-1, j+1] >= center) << 5
                code |= (img[i  , j+1] >= center) << 4
                code |= (img[i+1, j+1] >= center) << 3
                code |= (img[i+1, j  ] >= center) << 2
                code |= (img[i+1, j-1] >= center) << 1
                code |= (img[i  , j-1] >= center) << 0
                lbp[i-1, j-1] = code
        return lbp
    
    gray_small = cv2.resize(gray, (32, 32))
    lbp = calculate_lbp_fast(gray_small)
    lbp_hist = cv2.calcHist([lbp], [0], None, [26], [0, 256])
    lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
    features_dict['lbp'] = lbp_hist.astype(np.float32)
    print(f"  ✓ LBP (Local Binary Pattern): {len(lbp_hist)} features")
    print(f"    - Mean: {np.mean(lbp_hist):.6f}, Std: {np.std(lbp_hist):.6f}, Variance: {np.var(lbp_hist):.6f}")
    
    # ========== 3.3. EDGE FEATURES ==========
    print("\n3.3. EDGE FEATURES:")
    print("-" * 60)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
    features_dict['edge'] = edge_hist.astype(np.float32)
    print(f"  ✓ Canny Edge Histogram: {len(edge_hist)} features")
    print(f"    - Mean: {np.mean(edge_hist):.6f}, Std: {np.std(edge_hist):.6f}, Variance: {np.var(edge_hist):.6f}")
    print(f"    - Số pixel edge: {np.sum(edges > 0)} / {edges.size} ({100*np.sum(edges > 0)/edges.size:.2f}%)")
    
    # ========== 3.4. SIFT FEATURES ==========
    print("\n3.4. SIFT FEATURES:")
    print("-" * 60)
    
    sift = cv2.SIFT_create(nfeatures=50)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is None or len(descriptors) == 0:
        sift_feat = np.zeros(128)
        print(f"  ⚠ Không tìm thấy keypoints SIFT, sử dụng vector zero")
    else:
        sift_mean = np.mean(descriptors, axis=0)
        sift_std = np.std(descriptors, axis=0)
        sift_feat = np.concatenate([sift_mean, sift_std])[:128]
        print(f"  ✓ Tìm thấy {len(keypoints)} keypoints SIFT")
        print(f"    - Descriptors shape: {descriptors.shape}")
    
    features_dict['sift'] = sift_feat.astype(np.float32)
    print(f"  ✓ SIFT Features: {len(sift_feat)} features (mean + std của descriptors)")
    print(f"    - Mean: {np.mean(sift_feat):.6f}, Std: {np.std(sift_feat):.6f}, Variance: {np.var(sift_feat):.6f}")
    
    # Tổng hợp tất cả features
    all_features = np.concatenate([
        features_dict['rgb_histogram'],
        features_dict['hsv_histogram'],
        features_dict['lab_histogram'],
        features_dict['color_moments'],
        features_dict['color_ratios'],
        features_dict['hog'],
        features_dict['lbp'],
        features_dict['edge'],
        features_dict['sift']
    ])
    
    features_dict['all'] = all_features
    
    return features_dict

# Trích xuất features chi tiết
features_dict = extract_features_detailed(img_resized)

# Tổng hợp kết quả
print("\n" + "="*60)
print("TỔNG HỢP FEATURES:")
print("="*60)
total_features = 0
for feature_name, feature_array in features_dict.items():
    if feature_name != 'all':
        print(f"  {feature_name:20s}: {len(feature_array):4d} features")
        total_features += len(feature_array)

print(f"\n  {'TỔNG CỘNG':20s}: {total_features:4d} features")
print(f"  {'All features':20s}: {len(features_dict['all']):4d} features")

# Tính variance của từng loại feature
print("\n" + "="*60)
print("VARIANCE CỦA TỪNG LOẠI FEATURE:")
print("="*60)
for feature_name, feature_array in features_dict.items():
    if feature_name != 'all':
        variance = np.var(feature_array)
        print(f"  {feature_name:20s}: {variance:.6f}")

# Tính variance của ảnh
print("\n" + "="*60)
print("VARIANCE CỦA ẢNH:")
print("="*60)
variance_b = np.var(img_resized[:, :, 0])
variance_g = np.var(img_resized[:, :, 1])
variance_r = np.var(img_resized[:, :, 2])
variance_total = np.var(img_resized)
variance_features = np.var(features_dict['all'])

print(f"  Kênh B (Blue):     {variance_b:.2f}")
print(f"  Kênh G (Green):    {variance_g:.2f}")
print(f"  Kênh R (Red):      {variance_r:.2f}")
print(f"  Tổng thể (ảnh):    {variance_total:.2f}")
print(f"  Vector features:   {variance_features:.6f}")

# Hiển thị thông tin chi tiết về ảnh
print("\n" + "="*60)
print("THÔNG TIN CHI TIẾT VỀ ẢNH:")
print("="*60)
print(f"  Shape:              {img_resized.shape}")
print(f"  Dtype:              {img_resized.dtype}")
print(f"  Min pixel:           {img_resized.min()}")
print(f"  Max pixel:           {img_resized.max()}")
print(f"  Mean pixel:          {img_resized.mean():.2f}")
print(f"  Std pixel:           {img_resized.std():.2f}")

# ========== ÁP DỤNG PCA (từ V10_Balanced_Final.py) ==========
print("\n" + "="*60)
print("BƯỚC 4: ÁP DỤNG PCA (từ V10_Balanced_Final.py)")
print("="*60)

# Lấy features đầy đủ
all_features = features_dict['all'].reshape(1, -1).astype(np.float32)
print(f"\nFeatures trước PCA: {all_features.shape[1]} features")

# StandardScaler (giống V10)
print("\n4.1. StandardScaler (Chuẩn hóa features):")
print("-" * 60)
scaler = StandardScaler()
# Với 1 sample, fit_transform sẽ không scale đúng, nên ta chỉ transform
# Nhưng để demo, ta sẽ fit trên chính sample đó
features_scaled = scaler.fit_transform(all_features)
print(f"  ✓ Đã scale features")
print(f"    - Shape sau scale: {features_scaled.shape}")
print(f"    - Mean: {np.mean(features_scaled):.6f}, Std: {np.std(features_scaled):.6f}")
print(f"    - Variance: {np.var(features_scaled):.6f}")
print(f"    ⚠ Lưu ý: Với 1 sample, scaling không có ý nghĩa thống kê")

# PCA với variance=0.92 (giống V10)
pca_variance = 0.92
print(f"\n4.2. PCA (variance={pca_variance}):")
print("-" * 60)

# Lưu ý: PCA cần nhiều samples để fit đúng. Với 1 sample, ta sẽ:
# 1. Tạo thêm samples giả (bằng cách duplicate và thêm noise nhỏ)
# 2. Hoặc sử dụng PCA với số components cố định
# Ở đây ta sẽ dùng cách 1 để mô phỏng quá trình trong V10

use_gpu = False  # Có thể thay đổi thành True nếu có GPU

# Tạo dummy samples để fit PCA (giống như trong training)
# Trong thực tế, PCA được fit trên toàn bộ training set
print("  ⚠ Lưu ý: PCA cần nhiều samples để fit. Đang tạo dummy samples để demo...")
np.random.seed(42)  # Để kết quả reproducible
n_dummy_samples = min(100, features_scaled.shape[1] // 2)  # Tạo đủ samples để fit
dummy_features = np.tile(features_scaled, (n_dummy_samples, 1))
# Thêm noise nhỏ để tạo variation
noise = np.random.normal(0, 0.01, dummy_features.shape).astype(np.float32)
dummy_features = dummy_features + noise

if use_gpu and CUML_AVAILABLE:
    print("  Sử dụng cuML PCA (GPU)")
    n_comp = min(400, features_scaled.shape[1])
    pca_temp = cumlPCA(n_components=n_comp)
    pca_temp.fit(dummy_features.astype(np.float32))
    
    explained_var_ratio = pca_temp.explained_variance_ratio_
    if hasattr(explained_var_ratio, 'get'):
        explained_var_ratio = explained_var_ratio.get()
    
    cumsum = np.cumsum(explained_var_ratio)
    n_comp_opt = np.searchsorted(cumsum, pca_variance) + 1
    n_comp_opt = min(n_comp_opt, n_comp)
    
    pca = cumlPCA(n_components=n_comp_opt)
    pca.fit(dummy_features.astype(np.float32))
    features_pca = pca.transform(features_scaled.astype(np.float32))
    
    if hasattr(features_pca, 'get'):
        features_pca = features_pca.get()
else:
    print("  Sử dụng sklearn PCA (CPU)")
    # Với sklearn, ta có thể dùng n_components=variance
    pca = PCA(n_components=pca_variance)
    pca.fit(dummy_features)
    features_pca = pca.transform(features_scaled)

print(f"  ✓ PCA đã được fit trên {n_dummy_samples} dummy samples")
print(f"    - Số components: {features_pca.shape[1]}")
explained_var_sum = pca.explained_variance_ratio_.sum()
print(f"    - Explained variance ratio: {explained_var_sum:.4f} ({explained_var_sum*100:.2f}%)")
print(f"    - Shape sau PCA: {features_pca.shape}")

# Tính variance sau PCA
variance_pca = np.var(features_pca)
print(f"    - Variance sau PCA: {variance_pca:.6f}")
print(f"    - Mean: {np.mean(features_pca):.6f}, Std: {np.std(features_pca):.6f}")

# So sánh trước và sau PCA
print("\n" + "="*60)
print("SO SÁNH TRƯỚC VÀ SAU PCA:")
print("="*60)
print(f"  Số features trước PCA:     {all_features.shape[1]:4d}")
print(f"  Số features sau PCA:        {features_pca.shape[1]:4d}")
print(f"  Tỷ lệ giảm:                 {(1 - features_pca.shape[1]/all_features.shape[1])*100:.2f}%")
print(f"  Variance trước PCA:         {np.var(features_scaled):.6f}")
print(f"  Variance sau PCA:           {variance_pca:.6f}")
print(f"  Explained variance:         {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

# Hiển thị top components
print(f"\n  Top 10 components với variance cao nhất:")
explained_var = pca.explained_variance_ratio_
if hasattr(explained_var, 'get'):
    explained_var = explained_var.get()
top_indices = np.argsort(explained_var)[::-1][:10]
for i, idx in enumerate(top_indices, 1):
    print(f"    Component {idx:3d}: {explained_var[idx]:.6f} ({explained_var[idx]*100:.2f}%)")

# ========== TÍNH TOÁN MOMENTS THỐNG KÊ ==========
print("\n" + "="*60)
print("BƯỚC 5: TÍNH TOÁN MOMENTS THỐNG KÊ")
print("="*60)

def calculate_moments(data, name="Data"):
    """Tính toán các moment thống kê: Mean, Variance, Skewness, Kurtosis"""
    data_flat = data.flatten() if len(data.shape) > 1 else data
    
    mean_val = np.mean(data_flat)
    variance_val = np.var(data_flat)
    std_val = np.std(data_flat)
    skewness_val = skew(data_flat)
    kurtosis_val = kurtosis(data_flat)  # Excess kurtosis (kurtosis - 3)
    
    return {
        'mean': mean_val,
        'variance': variance_val,
        'std': std_val,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val
    }

print("\n5.1. MOMENTS CỦA ẢNH (theo kênh màu):")
print("-" * 60)
print(f"  {'Kênh':<10} {'Mean':<12} {'Variance':<12} {'Std':<12} {'Skewness':<12} {'Kurtosis':<12}")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

image_moments = {}
for i, channel_name in enumerate(['B (Blue)', 'G (Green)', 'R (Red)']):
    channel = img_resized[:, :, i]
    moments = calculate_moments(channel, channel_name)
    image_moments[channel_name] = moments
    print(f"  {channel_name:<10} {moments['mean']:>12.4f} {moments['variance']:>12.4f} "
          f"{moments['std']:>12.4f} {moments['skewness']:>12.4f} {moments['kurtosis']:>12.4f}")

# Moments tổng thể của ảnh
print("\n5.2. MOMENTS TỔNG THỂ CỦA ẢNH:")
print("-" * 60)
img_total_moments = calculate_moments(img_resized, "Ảnh tổng thể")
print(f"  Mean:      {img_total_moments['mean']:.6f}")
print(f"  Variance:  {img_total_moments['variance']:.6f}")
print(f"  Std:       {img_total_moments['std']:.6f}")
print(f"  Skewness:  {img_total_moments['skewness']:.6f}")
print(f"  Kurtosis:  {img_total_moments['kurtosis']:.6f}")

print("\n5.3. MOMENTS CỦA FEATURES (theo loại feature):")
print("-" * 60)
print(f"  {'Loại Feature':<25} {'Mean':<15} {'Variance':<15} {'Skewness':<15} {'Kurtosis':<15}")
print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

feature_moments = {}
for feature_name, feature_array in features_dict.items():
    if feature_name != 'all':
        moments = calculate_moments(feature_array, feature_name)
        feature_moments[feature_name] = moments
        print(f"  {feature_name:<25} {moments['mean']:>15.6f} {moments['variance']:>15.6f} "
              f"{moments['skewness']:>15.6f} {moments['kurtosis']:>15.6f}")

# Moments tổng thể của features
print("\n5.4. MOMENTS TỔNG THỂ CỦA FEATURES:")
print("-" * 60)
features_total_moments = calculate_moments(features_dict['all'], "Features tổng thể")
print(f"  Mean:      {features_total_moments['mean']:.6f}")
print(f"  Variance:  {features_total_moments['variance']:.6f}")
print(f"  Std:       {features_total_moments['std']:.6f}")
print(f"  Skewness:  {features_total_moments['skewness']:.6f}")
print(f"  Kurtosis:  {features_total_moments['kurtosis']:.6f}")

# Moments sau PCA
print("\n5.5. MOMENTS SAU PCA:")
print("-" * 60)
pca_moments = calculate_moments(features_pca, "Features sau PCA")
print(f"  Mean:      {pca_moments['mean']:.6f}")
print(f"  Variance:  {pca_moments['variance']:.6f}")
print(f"  Std:       {pca_moments['std']:.6f}")
print(f"  Skewness:  {pca_moments['skewness']:.6f}")
print(f"  Kurtosis:  {pca_moments['kurtosis']:.6f}")

# So sánh moments trước và sau PCA
print("\n5.6. SO SÁNH MOMENTS TRƯỚC VÀ SAU PCA:")
print("-" * 60)
print(f"  {'Moment':<15} {'Trước PCA':<20} {'Sau PCA':<20} {'Thay đổi':<15}")
print(f"  {'-'*15} {'-'*20} {'-'*20} {'-'*15}")
print(f"  {'Mean':<15} {features_total_moments['mean']:>20.6f} {pca_moments['mean']:>20.6f} "
      f"{pca_moments['mean'] - features_total_moments['mean']:>+15.6f}")
print(f"  {'Variance':<15} {features_total_moments['variance']:>20.6f} {pca_moments['variance']:>20.6f} "
      f"{pca_moments['variance'] - features_total_moments['variance']:>+15.6f}")
print(f"  {'Std':<15} {features_total_moments['std']:>20.6f} {pca_moments['std']:>20.6f} "
      f"{pca_moments['std'] - features_total_moments['std']:>+15.6f}")
print(f"  {'Skewness':<15} {features_total_moments['skewness']:>20.6f} {pca_moments['skewness']:>20.6f} "
      f"{pca_moments['skewness'] - features_total_moments['skewness']:>+15.6f}")
print(f"  {'Kurtosis':<15} {features_total_moments['kurtosis']:>20.6f} {pca_moments['kurtosis']:>20.6f} "
      f"{pca_moments['kurtosis'] - features_total_moments['kurtosis']:>+15.6f}")

# Vẽ biểu đồ moments
print("\n5.7. Vẽ biểu đồ so sánh moments...")
try:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Moments của ảnh theo kênh màu
    channels = ['Blue', 'Green', 'Red']
    channel_keys = ['B (Blue)', 'G (Green)', 'R (Red)']  # Đúng với key đã lưu
    x = np.arange(len(channels))
    width = 0.25
    
    means = [image_moments[key]['mean'] for key in channel_keys]
    variances = [image_moments[key]['variance'] for key in channel_keys]
    skewnesses = [image_moments[key]['skewness'] for key in channel_keys]
    kurtoses = [image_moments[key]['kurtosis'] for key in channel_keys]
    
    axes[0, 0].bar(x - width*1.5, means, width, label='Mean', color='#3498db', alpha=0.8)
    axes[0, 0].bar(x - width*0.5, variances, width, label='Variance', color='#2ecc71', alpha=0.8)
    axes[0, 0].bar(x + width*0.5, skewnesses, width, label='Skewness', color='#e74c3c', alpha=0.8)
    axes[0, 0].bar(x + width*1.5, kurtoses, width, label='Kurtosis', color='#f39c12', alpha=0.8)
    axes[0, 0].set_xlabel('Kênh màu', fontsize=11)
    axes[0, 0].set_ylabel('Giá trị', fontsize=11)
    axes[0, 0].set_title('Moments của ảnh theo kênh màu', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(channels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. So sánh moments của features
    feature_names = [name for name in features_dict.keys() if name != 'all']
    n_features = len(feature_names)
    x2 = np.arange(n_features)
    
    feat_means = [feature_moments[name]['mean'] for name in feature_names]
    feat_vars = [feature_moments[name]['variance'] for name in feature_names]
    feat_skews = [feature_moments[name]['skewness'] for name in feature_names]
    feat_kurts = [feature_moments[name]['kurtosis'] for name in feature_names]
    
    axes[0, 1].bar(x2 - width*1.5, feat_means, width, label='Mean', color='#3498db', alpha=0.8)
    axes[0, 1].bar(x2 - width*0.5, feat_vars, width, label='Variance', color='#2ecc71', alpha=0.8)
    axes[0, 1].bar(x2 + width*0.5, feat_skews, width, label='Skewness', color='#e74c3c', alpha=0.8)
    axes[0, 1].bar(x2 + width*1.5, feat_kurts, width, label='Kurtosis', color='#f39c12', alpha=0.8)
    axes[0, 1].set_xlabel('Loại Feature', fontsize=11)
    axes[0, 1].set_ylabel('Giá trị', fontsize=11)
    axes[0, 1].set_title('Moments của Features', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x2)
    axes[0, 1].set_xticklabels([name[:10] for name in feature_names], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. So sánh Mean, Variance, Std trước và sau PCA
    moments_comparison = ['Mean', 'Variance', 'Std']
    before_pca = [features_total_moments['mean'], features_total_moments['variance'], features_total_moments['std']]
    after_pca = [pca_moments['mean'], pca_moments['variance'], pca_moments['std']]
    
    x3 = np.arange(len(moments_comparison))
    axes[0, 2].bar(x3 - width/2, before_pca, width, label='Trước PCA', color='#3498db', alpha=0.8)
    axes[0, 2].bar(x3 + width/2, after_pca, width, label='Sau PCA', color='#e74c3c', alpha=0.8)
    axes[0, 2].set_xlabel('Moment', fontsize=11)
    axes[0, 2].set_ylabel('Giá trị', fontsize=11)
    axes[0, 2].set_title('So sánh Moments trước/sau PCA', fontsize=12, fontweight='bold')
    axes[0, 2].set_xticks(x3)
    axes[0, 2].set_xticklabels(moments_comparison)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Skewness và Kurtosis theo kênh màu
    axes[1, 0].bar(x - width/2, skewnesses, width, label='Skewness', color='#e74c3c', alpha=0.8)
    axes[1, 0].bar(x + width/2, kurtoses, width, label='Kurtosis', color='#f39c12', alpha=0.8)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Kênh màu', fontsize=11)
    axes[1, 0].set_ylabel('Giá trị', fontsize=11)
    axes[1, 0].set_title('Skewness và Kurtosis theo kênh màu', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(channels)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Skewness và Kurtosis của features
    axes[1, 1].bar(x2 - width/2, feat_skews, width, label='Skewness', color='#e74c3c', alpha=0.8)
    axes[1, 1].bar(x2 + width/2, feat_kurts, width, label='Kurtosis', color='#f39c12', alpha=0.8)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Loại Feature', fontsize=11)
    axes[1, 1].set_ylabel('Giá trị', fontsize=11)
    axes[1, 1].set_title('Skewness và Kurtosis của Features', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels([name[:10] for name in feature_names], rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. So sánh Skewness và Kurtosis trước/sau PCA
    skew_kurt_comparison = ['Skewness', 'Kurtosis']
    before_skew_kurt = [features_total_moments['skewness'], features_total_moments['kurtosis']]
    after_skew_kurt = [pca_moments['skewness'], pca_moments['kurtosis']]
    
    x4 = np.arange(len(skew_kurt_comparison))
    axes[1, 2].bar(x4 - width/2, before_skew_kurt, width, label='Trước PCA', color='#3498db', alpha=0.8)
    axes[1, 2].bar(x4 + width/2, after_skew_kurt, width, label='Sau PCA', color='#e74c3c', alpha=0.8)
    axes[1, 2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 2].set_xlabel('Moment', fontsize=11)
    axes[1, 2].set_ylabel('Giá trị', fontsize=11)
    axes[1, 2].set_title('Skewness & Kurtosis trước/sau PCA', fontsize=12, fontweight='bold')
    axes[1, 2].set_xticks(x4)
    axes[1, 2].set_xticklabels(skew_kurt_comparison)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Phân tích Moments thống kê', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('comparison_moments.png', dpi=150, bbox_inches='tight')
    print("  ✓ Đã lưu: comparison_moments.png")
    plt.close()
    
except Exception as e:
    print(f"  ⚠ Lỗi khi vẽ biểu đồ moments: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("HOÀN TẤT TÍNH TOÁN MOMENTS")
print("="*60)

# ========== TẠO BÁO CÁO TỔNG HỢP ==========
print("\n" + "="*60)
print("BƯỚC 6: TẠO BÁO CÁO TỔNG HỢP")
print("="*60)

def image_to_base64(image_path):
    """Chuyển ảnh sang base64 để embed vào HTML"""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
            return base64.b64encode(img_data).decode('utf-8')
    except:
        return None

def create_html_report():
    """Tạo báo cáo HTML đầy đủ"""
    print("\n6.1. Tạo báo cáo HTML...")
    
    # Thu thập dữ liệu
    report_data = {
        'image_path': image_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_size': f"{original_width}x{original_height}",
        'resized_size': f"{resized_width}x{resized_height}",
        'total_features': len(features_dict['all']),
        'features_after_pca': features_pca.shape[1],
        'pca_reduction': f"{(1 - features_pca.shape[1]/len(features_dict['all']))*100:.2f}%",
        'pca_variance': f"{pca.explained_variance_ratio_.sum()*100:.2f}%"
    }
    
    # Tạo HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Phân Tích Ảnh - {Path(image_path).name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 25px;
            background: #fafafa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .section h3 {{
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .metric-card {{
            display: inline-block;
            background: white;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-width: 200px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-item {{
            text-align: center;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .image-item p {{
            margin-top: 10px;
            font-weight: bold;
            color: #667eea;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .summary-box h3 {{
            color: white;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .summary-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        .summary-box li {{
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .summary-box li:before {{
            content: "✓ ";
            font-weight: bold;
            margin-right: 10px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #ddd;
            margin-top: 40px;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
            margin: 5px;
        }}
        .badge-success {{
            background: #2ecc71;
            color: white;
        }}
        .badge-info {{
            background: #3498db;
            color: white;
        }}
        .badge-warning {{
            background: #f39c12;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Báo Cáo Phân Tích Ảnh</h1>
            <p>Ảnh: {Path(image_path).name} | Thời gian: {report_data['timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>📋 Tóm Tắt Executive</h2>
            <div class="summary-box">
                <h3>Thông Tin Cơ Bản</h3>
                <ul>
                    <li>Kích thước gốc: {report_data['original_size']} pixels</li>
                    <li>Kích thước sau xử lý: {report_data['resized_size']} pixels</li>
                    <li>Tổng số features: {report_data['total_features']} features</li>
                    <li>Features sau PCA: {report_data['features_after_pca']} features</li>
                    <li>Giảm chiều: {report_data['pca_reduction']}</li>
                    <li>Variance giữ lại: {report_data['pca_variance']}</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>🖼️ So Sánh Ảnh</h2>
            <div class="image-grid">
                <div class="image-item">
                    <img src="comparison_images.png" alt="So sánh ảnh" style="max-width: 100%;">
                    <p>So sánh ảnh trước và sau xử lý</p>
                </div>
            </div>
            <table>
                <tr>
                    <th>Thông số</th>
                    <th>Trước xử lý</th>
                    <th>Sau resize (256×256)</th>
                    <th>Thay đổi</th>
                </tr>
                <tr>
                    <td>Chiều rộng</td>
                    <td>{original_width} px</td>
                    <td>{resized_width} px</td>
                    <td>{resized_width - original_width:+d} px</td>
                </tr>
                <tr>
                    <td>Chiều cao</td>
                    <td>{original_height} px</td>
                    <td>{resized_height} px</td>
                    <td>{resized_height - original_height:+d} px</td>
                </tr>
                <tr>
                    <td>Tổng số pixel</td>
                    <td>{original_size:,} px</td>
                    <td>{resized_size:,} px</td>
                    <td>{(resized_size/original_size - 1)*100:+.2f}%</td>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{orig_total_mean:.2f}</td>
                    <td>{resized_total_mean:.2f}</td>
                    <td>{resized_total_mean - orig_total_mean:+.2f}</td>
                </tr>
                <tr>
                    <td>Variance</td>
                    <td>{orig_total_var:.2f}</td>
                    <td>{resized_total_var:.2f}</td>
                    <td>{resized_total_var - orig_total_var:+.2f}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🎨 Phân Tích Features</h2>
            <h3>Chi Tiết Features</h3>
            <table>
                <tr>
                    <th>Loại Feature</th>
                    <th>Số lượng</th>
                    <th>Mean</th>
                    <th>Variance</th>
                    <th>Skewness</th>
                    <th>Kurtosis</th>
                </tr>
"""
    
    # Thêm features vào bảng
    for feature_name, feature_array in features_dict.items():
        if feature_name != 'all':
            moments = feature_moments.get(feature_name, {})
            html_content += f"""
                <tr>
                    <td><strong>{feature_name.replace('_', ' ').title()}</strong></td>
                    <td>{len(feature_array)}</td>
                    <td>{moments.get('mean', 0):.6f}</td>
                    <td>{moments.get('variance', 0):.6f}</td>
                    <td>{moments.get('skewness', 0):.6f}</td>
                    <td>{moments.get('kurtosis', 0):.6f}</td>
                </tr>
"""
    
    html_content += f"""
            </table>
            
            <h3>Metrics Tổng Hợp</h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: center;">
                <div class="metric-card">
                    <div class="metric-value">{len(features_dict['all'])}</div>
                    <div class="metric-label">Tổng Features</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{features_pca.shape[1]}</div>
                    <div class="metric-label">Features sau PCA</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{features_total_moments['variance']:.6f}</div>
                    <div class="metric-label">Variance Features</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{pca_moments['variance']:.6f}</div>
                    <div class="metric-label">Variance sau PCA</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Biểu Đồ Phân Tích</h2>
            <div class="image-grid">
                <div class="image-item">
                    <img src="comparison_statistics.png" alt="Thống kê" style="max-width: 100%;">
                    <p>So sánh thống kê theo kênh màu</p>
                </div>
                <div class="image-item">
                    <img src="comparison_histogram.png" alt="Histogram" style="max-width: 100%;">
                    <p>Phân bố pixel theo kênh màu</p>
                </div>
                <div class="image-item">
                    <img src="comparison_overview.png" alt="Tổng quan" style="max-width: 100%;">
                    <p>So sánh tổng quan</p>
                </div>
                <div class="image-item">
                    <img src="comparison_moments.png" alt="Moments" style="max-width: 100%;">
                    <p>Phân tích moments thống kê</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔬 Phân Tích PCA</h2>
            <table>
                <tr>
                    <th>Thông số</th>
                    <th>Trước PCA</th>
                    <th>Sau PCA</th>
                    <th>Thay đổi</th>
                </tr>
                <tr>
                    <td>Số features</td>
                    <td>{len(features_dict['all'])}</td>
                    <td>{features_pca.shape[1]}</td>
                    <td><span class="badge badge-success">-{report_data['pca_reduction']}</span></td>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{features_total_moments['mean']:.6f}</td>
                    <td>{pca_moments['mean']:.6f}</td>
                    <td>{pca_moments['mean'] - features_total_moments['mean']:+.6f}</td>
                </tr>
                <tr>
                    <td>Variance</td>
                    <td>{features_total_moments['variance']:.6f}</td>
                    <td>{pca_moments['variance']:.6f}</td>
                    <td>{pca_moments['variance'] - features_total_moments['variance']:+.6f}</td>
                </tr>
                <tr>
                    <td>Skewness</td>
                    <td>{features_total_moments['skewness']:.6f}</td>
                    <td>{pca_moments['skewness']:.6f}</td>
                    <td>{pca_moments['skewness'] - features_total_moments['skewness']:+.6f}</td>
                </tr>
                <tr>
                    <td>Kurtosis</td>
                    <td>{features_total_moments['kurtosis']:.6f}</td>
                    <td>{pca_moments['kurtosis']:.6f}</td>
                    <td>{pca_moments['kurtosis'] - features_total_moments['kurtosis']:+.6f}</td>
                </tr>
                <tr>
                    <td>Explained Variance</td>
                    <td>100%</td>
                    <td>{report_data['pca_variance']}</td>
                    <td><span class="badge badge-info">Giữ lại {report_data['pca_variance']}</span></td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>💡 Kết Luận và Insights</h2>
            <div class="summary-box">
                <h3>Nhận Xét Chính</h3>
                <ul>
                    <li>Ảnh đã được resize từ {original_width}×{original_height} về 256×256 để chuẩn hóa</li>
                    <li>Trích xuất được {len(features_dict['all'])} features từ nhiều nguồn (Color, Texture, Edge, SIFT)</li>
                    <li>PCA giảm từ {len(features_dict['all'])} xuống {features_pca.shape[1]} features, giữ lại {report_data['pca_variance']} variance</li>
                    <li>Variance của features: {features_total_moments['variance']:.6f} (trước PCA) → {pca_moments['variance']:.6f} (sau PCA)</li>
                    <li>Skewness: {features_total_moments['skewness']:.4f} cho thấy phân bố {'lệch phải' if features_total_moments['skewness'] > 0 else 'lệch trái' if features_total_moments['skewness'] < 0 else 'đối xứng'}</li>
                    <li>Kurtosis: {features_total_moments['kurtosis']:.4f} cho thấy phân bố {'nhọn hơn' if features_total_moments['kurtosis'] > 0 else 'phẳng hơn' if features_total_moments['kurtosis'] < 0 else 'bình thường'} phân bố chuẩn</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Báo cáo được tạo tự động bởi Image Analysis Tool</p>
            <p>Thời gian: {report_data['timestamp']}</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Lưu file HTML
    html_file = 'image_analysis_report.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"  ✓ Đã tạo báo cáo HTML: {html_file}")
    return html_file

def export_to_json():
    """Export dữ liệu ra JSON"""
    print("6.2. Export dữ liệu ra JSON...")
    
    export_data = {
        'image_info': {
            'path': image_path,
            'original_size': {'width': int(original_width), 'height': int(original_height)},
            'resized_size': {'width': int(resized_width), 'height': int(resized_height)},
            'original_stats': {
                'mean': float(orig_total_mean),
                'std': float(orig_total_std),
                'variance': float(orig_total_var),
                'min': int(orig_total_min),
                'max': int(orig_total_max)
            },
            'resized_stats': {
                'mean': float(resized_total_mean),
                'std': float(resized_total_std),
                'variance': float(resized_total_var),
                'min': int(resized_total_min),
                'max': int(resized_total_max)
            }
        },
        'features': {
            'total_count': int(len(features_dict['all'])),
            'after_pca': int(features_pca.shape[1]),
            'pca_reduction_percent': float((1 - features_pca.shape[1]/len(features_dict['all']))*100),
            'by_type': {}
        },
        'pca': {
            'variance_retained': float(pca.explained_variance_ratio_.sum()),
            'components': int(features_pca.shape[1]),
            'moments': {
                'mean': float(pca_moments['mean']),
                'variance': float(pca_moments['variance']),
                'std': float(pca_moments['std']),
                'skewness': float(pca_moments['skewness']),
                'kurtosis': float(pca_moments['kurtosis'])
            }
        },
        'moments': {
            'image_total': {
                'mean': float(img_total_moments['mean']),
                'variance': float(img_total_moments['variance']),
                'std': float(img_total_moments['std']),
                'skewness': float(img_total_moments['skewness']),
                'kurtosis': float(img_total_moments['kurtosis'])
            },
            'features_total': {
                'mean': float(features_total_moments['mean']),
                'variance': float(features_total_moments['variance']),
                'std': float(features_total_moments['std']),
                'skewness': float(features_total_moments['skewness']),
                'kurtosis': float(features_total_moments['kurtosis'])
            },
            'by_channel': {},
            'by_feature_type': {}
        }
    }
    
    # Thêm moments theo kênh màu
    for key, moments in image_moments.items():
        export_data['moments']['by_channel'][key] = {
            'mean': float(moments['mean']),
            'variance': float(moments['variance']),
            'std': float(moments['std']),
            'skewness': float(moments['skewness']),
            'kurtosis': float(moments['kurtosis'])
        }
    
    # Thêm features theo loại
    for feature_name, feature_array in features_dict.items():
        if feature_name != 'all':
            export_data['features']['by_type'][feature_name] = {
                'count': int(len(feature_array)),
                'moments': {
                    'mean': float(feature_moments[feature_name]['mean']),
                    'variance': float(feature_moments[feature_name]['variance']),
                    'std': float(feature_moments[feature_name]['std']),
                    'skewness': float(feature_moments[feature_name]['skewness']),
                    'kurtosis': float(feature_moments[feature_name]['kurtosis'])
                }
            }
    
    json_file = 'image_analysis_data.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Đã export JSON: {json_file}")
    return json_file

def create_summary_table():
    """Tạo bảng tóm tắt cho báo cáo"""
    print("6.3. Tạo bảng tóm tắt...")
    
    summary_file = 'summary_table.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BẢNG TÓM TẮT PHÂN TÍCH ẢNH\n")
        f.write("="*80 + "\n\n")
        f.write(f"Ảnh: {Path(image_path).name}\n")
        f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. THÔNG TIN ẢNH\n")
        f.write("-"*80 + "\n")
        f.write(f"  Kích thước gốc:     {original_width}×{original_height} pixels\n")
        f.write(f"  Kích thước sau:     {resized_width}×{resized_height} pixels\n")
        f.write(f"  Tổng số pixel:      {original_size:,} → {resized_size:,}\n")
        f.write(f"  Mean:               {orig_total_mean:.2f} → {resized_total_mean:.2f}\n")
        f.write(f"  Variance:           {orig_total_var:.2f} → {resized_total_var:.2f}\n\n")
        
        f.write("2. FEATURES\n")
        f.write("-"*80 + "\n")
        f.write(f"  Tổng số features:   {len(features_dict['all'])}\n")
        f.write(f"  Sau PCA:            {features_pca.shape[1]}\n")
        f.write(f"  Giảm:               {(1 - features_pca.shape[1]/len(features_dict['all']))*100:.2f}%\n")
        f.write(f"  Variance giữ lại:   {pca.explained_variance_ratio_.sum()*100:.2f}%\n\n")
        
        f.write("3. MOMENTS THỐNG KÊ\n")
        f.write("-"*80 + "\n")
        f.write(f"  Mean:               {features_total_moments['mean']:.6f}\n")
        f.write(f"  Variance:           {features_total_moments['variance']:.6f}\n")
        f.write(f"  Std:                {features_total_moments['std']:.6f}\n")
        f.write(f"  Skewness:           {features_total_moments['skewness']:.6f}\n")
        f.write(f"  Kurtosis:           {features_total_moments['kurtosis']:.6f}\n\n")
        
        f.write("4. CHI TIẾT FEATURES\n")
        f.write("-"*80 + "\n")
        for feature_name, feature_array in features_dict.items():
            if feature_name != 'all':
                moments = feature_moments.get(feature_name, {})
                f.write(f"  {feature_name:20s}: {len(feature_array):4d} features, "
                       f"variance={moments.get('variance', 0):.6f}\n")
    
    print(f"  ✓ Đã tạo bảng tóm tắt: {summary_file}")
    return summary_file

# Tạo các báo cáo
try:
    html_file = create_html_report()
    json_file = export_to_json()
    summary_file = create_summary_table()
    
    print("\n" + "="*60)
    print("✓ ĐÃ TẠO TẤT CẢ BÁO CÁO")
    print("="*60)
    print(f"  📄 HTML Report:     {html_file}")
    print(f"  📊 JSON Data:       {json_file}")
    print(f"  📋 Summary Table:   {summary_file}")
    print(f"  🖼️  Visualizations:  comparison_*.png (4 files)")
    print("\n💡 Mở file HTML trong trình duyệt để xem báo cáo đầy đủ!")
    
except Exception as e:
    print(f"⚠ Lỗi khi tạo báo cáo: {e}")
    import traceback
    traceback.print_exc()

