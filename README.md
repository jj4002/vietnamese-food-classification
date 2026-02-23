# 🍜 Phân loại món ăn Việt Nam (Vietnamese Food Classification)


## 📸 Các món ăn hỗ trợ phân loại
Hệ thống hiện tại có thể nhận diện:
*   **Bánh chưng, Bánh mì, Bánh xèo, Bún bò Huế, Bún đậu mắm tôm, Chả giò, Cháo lòng.**

---

## �️ Cài đặt môi trường

Yêu cầu: **Python 3.9+**

1.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

2.  (Tùy chọn) Cài đặt riêng cho backend:
    ```bash
    pip install -r backend/requirements.txt
    ```

---

## 🏗️ Cấu trúc dự án
*   **`dataset/`**: Chứa dữ liệu ảnh thu thập được (bản zip và thư mục giải nén).
*   **`analyze/`**: Chứa script `analyze_image.py` dùng để phân tích các đặc trưng (Histogram, HOG, LBP, Edge, SIFT) trước khi huấn luyện.
*   **`crawl_image/`**: Bộ công cụ thu thập dữ liệu tự động từ Flickr và iStockphoto.
*   **`training/`**: Chứa các phiên bản mô hình từ V1 đến V10.
*   **`models/`**: Lưu trữ model đã được đóng gói (`.pkl`).
*   **`backend/`**: API xử lý nhận diện (FastAPI).
*   **`frontend/`**: Giao diện người dùng đơn giản (HTML/CSS/JS).

---

## 🔬 Các phiên bản mô hình (Model Versions)

Thư mục `training/` chứa quá trình phát triển mô hình qua nhiều phiên bản, từ đơn giản đến phức tạp. Phiên bản chính được sử dụng trong production là **V10_CPU**.

---

### V1 – Baseline (`training/V1/V1_Baseline.py`)
Phiên bản đầu tiên, đặt nền tảng cho toàn bộ pipeline.

| Đặc điểm | Chi tiết |
|---|---|
| **Mô hình** | SVM kernel RBF (single model) |
| **Đặc trưng** | RGB/HSV/LAB histogram, HOG, LBP, Canny edges, Sobel, SIFT (256 chiều) |
| **Cân bằng dữ liệu** | SMOTE / ADASYN / SMOTETomek |
| **Giảm chiều** | PCA (350 components) |
| **Kích thước ảnh** | 256×256 |
| **Tăng cường dữ liệu** | Flip, tăng/giảm độ sáng, xoay nhẹ (tối đa 5 bản/ảnh) |
| **GPU** | Hỗ trợ cuML (tự động fallback về scikit-learn nếu không có GPU) |

> **Mục tiêu**: Xây dựng pipeline cơ bản, kiểm tra tính khả thi của hướng tiếp cận trích đặc trưng thủ công + SVM.

---

### V4 – Ensemble (`training/V4_Ensemble.py`)
Thử nghiệm kết hợp nhiều mô hình ML để cải thiện độ chính xác.

| Đặc điểm | Chi tiết |
|---|---|
| **Mô hình** | VotingClassifier (soft voting) gồm 7 mô hình: SVM RBF, SVM Linear, Random Forest, Extra Trees, XGBoost/Gradient Boosting, KNN, Logistic Regression |
| **Đặc trưng** | RGB/HSV/LAB histogram, HOG, Sobel, Canny, Laplacian, thống kê màu sắc, phân tích 4 vùng ảnh |
| **Cân bằng dữ liệu** | SMOTE |
| **Giảm chiều** | PCA (300 components) |
| **Kích thước ảnh** | 224×224 |
| **Tăng cường dữ liệu** | Flip, tăng/giảm độ sáng (tối đa 5 bản/ảnh) |

> **Mục tiêu**: Khám phá xem ensemble nhiều thuật toán ML có vượt trội so với SVM đơn lẻ không.

---

### V10 – Balanced Final GPU (`training/V10/V10_Balanced_Final.py`)
Phiên bản tối ưu cho môi trường có GPU, tập trung vào chống overfitting.

| Đặc điểm | Chi tiết |
|---|---|
| **Mô hình** | Ensemble 3 SVM RBF độc lập (majority voting) |
| **Đặc trưng** | RGB/HSV/LAB histogram, HOG (64×64), LBP tối ưu, Canny, SIFT (128 chiều) |
| **Cân bằng dữ liệu** | Augmentation theo tỉ lệ class (không dùng SMOTE) |
| **Giảm chiều** | PCA giữ 92% phương sai (~653 components) |
| **Kích thước ảnh** | 256×256 |
| **Kỹ thuật chống overfitting** | Feature noise (σ=0.05), Feature dropout (5%), Subsampling (85%) cho mỗi model trong ensemble |
| **GPU** | Ưu tiên cuML (NVIDIA), fallback CPU |
| **Siêu tham số** | C=0.3, n_ensemble=3 |

> **Mục tiêu**: Đạt accuracy cao nhất có thể trên GPU, đây là bản tham chiếu để so sánh.

---

### ⭐ V10_CPU – Phiên bản chính (`training/V10_CPU/v10_balanced_cpu.ipynb`)
**Đây là phiên bản được sử dụng trong production**, được thiết kế để chạy hoàn toàn trên CPU (Google Colab hoặc bất kỳ máy nào không có GPU).

| Đặc điểm | Chi tiết |
|---|---|
| **Mô hình** | Ensemble 3 SVM RBF (sklearn, CPU only) |
| **Đặc trưng** | Giống V10 GPU (RGB/HSV/LAB histogram, HOG, LBP, Canny, SIFT 128 chiều) |
| **Cân bằng dữ liệu** | Augmentation theo tỉ lệ class |
| **Giảm chiều** | PCA (92% variance → 653 components) |
| **Kỹ thuật chống overfitting** | Feature noise, Feature dropout, Subsampling |
| **GPU** | Không cần – chỉ dùng `scikit-learn` `SVC` |
| **Môi trường huấn luyện** | Google Colab (CPU runtime) |
| **Thời gian train** | ~36 phút trên Colab CPU |

**Kết quả thực tế (chạy trên 6.052 ảnh, 7 lớp):**

| Tập | Accuracy | F1-score |
|---|---|---|
| Train | 85.10% | 85.06% |
| Validation | 67.73% | 67.64% |
| **Test** | **70.15%** | **70.16%** |

**Cách chạy:**
```bash
# Mở notebook trên Google Colab (CPU)
training/V10_CPU/v10_balanced_cpu.ipynb
```
hoặc dùng script Python:
```bash
python training/V10_CPU/v10_balanced_cpu.py
```

> Model đã huấn luyện được lưu tại `models/v10_balanced.pkl` và được backend tải lên khi khởi động.

---

## 🧪 Quy trình thực hiện

### 1. Thu thập dữ liệu
Nếu bạn muốn mở rộng dataset:
```bash
cd crawl_image
# Thu thập từ Flickr
python crawl_flickr.py
# Hoặc iStockphoto
python crawl_istockphoto.py
```

### 2. Phân tích đặc trưng (Feature Extraction)
Đây là bước quan trọng để hiểu dữ liệu trước khi train:
```bash
python analyze/analyze_image.py
```
Script này sẽ trích xuất và so sánh: Color Features (RGB, HSV, LAB), Texture (LBP, HOG), Shape (SIFT),...

### 3. Huấn luyện mô hình
Project sử dụng mô hình **V10 Balanced Final (SVM Ensemble)** để đạt kết quả tốt nhất và tránh Overfitting:
```bash
python training/V10/V10_Balanced_Final.py
```
hoặc bản không có **cuml (gpu)**
```bash
python training/V10_CPU/v10_balanced_cpu.py
``` 


---

## � Chạy ứng dụng

### Khởi động API (Backend)
```bash
cd backend
python run_server.py
```
API sẽ chạy tại `http://127.0.0.1:8000`. Bạn có thể xem tài liệu API tại `http://127.0.0.1:8000/docs`.

### Giao diện người dùng (Frontend)
Mở trực tiếp file `frontend/index.html` trong trình duyệt để sử dụng giao diện web, tải ảnh lên và nhận kết quả phân loại.

---

## � Kết quả đạt được
Mô hình **V10** đạt độ chính xác khoảng **71%** trên tập kiểm thử (Test set), với sự kết hợp của nhiều loại đặc trưng và kỹ thuật Ensemble 3 mô hình SVM độc lập.

---

## 📜 License
Dự án được phát hành dưới bản quyền [MIT License](LICENSE).

---
*Thực hiện bởi nhóm sinh viên phục vụ cho môn Machine Learning tại trường, không sử dụng cho mục đích thương mại.*