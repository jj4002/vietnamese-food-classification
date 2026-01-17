# Vietnamese Food Classification

Đồ án môn Machine Learning PTITHCM. Sử dụng Machine Learning để phân loại các món ăn Việt Nam từ hình ảnh.

## Các món ăn được phân loại

- Bánh chưng
- Bánh mì
- Bánh xèo
- Bún bò Huế
- Bún đậu mắm tôm
- Chả giò
- Cháo lòng

## Cấu trúc dự án

```
vietnamese-food-classification/
├── dataset/
│   └── Train/           # Dữ liệu huấn luyện
├── crawl_image/         # Script thu thập dữ liệu
├── train/               # Mô hình huấn luyện
├── analyze_image.py     # Phân tích và trích xuất đặc trưng ảnh
├── requirements.txt     # Các thư viện cần thiết
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Các phiên bản mô hình

### V1 - Baseline
Sử dụng SVM RBF với SMOTE để cân bằng dữ liệu.

### V4 - Ensemble
Kết hợp nhiều mô hình Machine Learning:
- SVM RBF
- SVM Linear
- Random Forest
- Extra Trees
- XGBoost/Gradient Boosting
- KNN
- Logistic Regression

### V10 - Balanced Final (Mô hình tốt nhất)
SVM Ensemble với các kỹ thuật chống overfitting:
- Feature noise
- Feature dropout
- Subsampling
- Ensemble 3 mô hình SVM

## Huấn luyện mô hình

```bash
# V1 - Baseline
python train/V1_Baseline.py

# V4 - Ensemble
python train/V4_Ensemble.py

# V10 - Balanced Final (khuyên dùng)
python train/V10_Balanced_Final.py
```

## Sử dụng mô hình

```python
import pickle
import cv2

# Load mô hình
with open('v10_balanced.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Dự đoán
classifier = FoodClassifierSVM_Balanced('dataset/Train')
classifier.load_model('v10_balanced.pkl')
predicted_class, confidence = classifier.predict('path/to/image.jpg')

print(f"Predicted: {predicted_class} with {confidence:.1f}% confidence")
```

## Thu thập dữ liệu

```bash
cd crawl_image

# Crawl từ Flickr
python crawl_flickr.py

# Crawl từ iStockphoto
python crawl_istockphoto.py

# Tải ảnh từ CSV
python download_images.py
```

## Phân tích ảnh

```bash
python analyze_image.py
```

Điều này sẽ tạo báo cáo chi tiết về:
- Kích thước ảnh trước/sau khi resize
- Thống kê pixel theo từng kênh màu
- Các đặc trưng (histogram, HOG, LBP, Edge, SIFT)
- Biểu đồ so sánh trực quan

## Các đặc trưng được sử dụng

### Color Features
- RGB Histogram (32 bins)
- HSV Histogram (32 bins)
- LAB Histogram (24 bins)
- Color Moments (Mean, Std, Skewness)
- Color Ratios

### Texture Features
- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Pattern)
- Edge Histogram (Canny)
- Sobel/Laplacian gradients

### Shape Features
- SIFT keypoints và descriptors

## Kết quả

Mô hình V10 Balanced đạt được kết quả tốt nhất với:
- Accuracy trên tập test: ~71%
- Khoảng cách giữa train và test: ~13%

## Yêu cầu hệ thống

- Python 3.7+
- 8GB RAM
- GPU hỗ trợ CUDA (tùy chọn, để huấn luyện nhanh hơn)

## Thư viện chính

- scikit-learn - Machine Learning
- opencv-python - Xử lý ảnh
- numpy - Tính toán
- pandas - Xử lý dữ liệu
- matplotlib/seaborn - Trực quan hóa
- imbalanced-learn - Cân bằng dữ liệu
- requests/undetected-playwright - Web scraping

## License

MIT License