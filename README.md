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