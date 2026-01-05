import csv
import os
from pathlib import Path
from typing import List, Optional

import requests
from requests import Response

# Cấu hình mặc định
CSV_DIR = Path(__file__).parent / "csv"
IMAGES_DIR = Path(__file__).parent / "images"
REQUEST_TIMEOUT = 30  # seconds
DELAY_BETWEEN_DOWNLOADS = 0.5  # seconds


def read_image_urls(csv_path: Path) -> List[str]:
    """Đọc danh sách URL từ CSV file"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    urls: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0]:
                urls.append(row[0].strip())

    if not urls:
        raise ValueError("Không tìm thấy URL nào trong CSV.")

    return urls


def get_image_extension(url: str, content_type: Optional[str] = None) -> str:
    """Xác định extension từ URL hoặc Content-Type"""
    # Thử lấy từ URL trước
    path = Path(url.split("?")[0])  # Bỏ query string
    suffix = path.suffix.lower()

    # Nếu có suffix hợp lệ (4 ký tự trở xuống)
    if suffix and len(suffix) <= 5 and suffix in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]:
        return suffix

    # Nếu không có, thử từ Content-Type
    if content_type:
        content_type = content_type.lower()
        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg"
        elif "png" in content_type:
            return ".png"
        elif "gif" in content_type:
            return ".gif"
        elif "webp" in content_type:
            return ".webp"
        elif "bmp" in content_type:
            return ".bmp"

    # Mặc định là .jpg
    return ".jpg"


def download_image(url: str, target_path: Path, headers: Optional[dict] = None) -> bool:
    """Tải ảnh từ URL và lưu vào đường dẫn"""
    try:
        response: Response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Kiểm tra xem có phải là ảnh không
        content_type = response.headers.get("Content-Type", "")
        if content_type and "image" not in content_type:
            print(f"  ⚠️  Không phải ảnh: {content_type}")
            return False

        # Đảm bảo extension đúng
        if not target_path.suffix:
            ext = get_image_extension(url, content_type)
            target_path = target_path.with_suffix(ext)

        target_path.write_bytes(response.content)
        return True
    except Exception as e:
        print(f"  ❌ Lỗi tải ảnh: {e}")
        return False


def download_images_from_csv(
    csv_path: Path,
    output_dir: Path,
    start_index: int = 1,
    skip_existing: bool = True,
    delay: float = DELAY_BETWEEN_DOWNLOADS,
    headers: Optional[dict] = None,
) -> None:
    """Download tất cả ảnh từ CSV file"""
    urls = read_image_urls(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Bắt đầu tải {len(urls)} ảnh từ {csv_path}")
    print(f"Lưu vào thư mục: {output_dir}\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for index, url in enumerate(urls, start=start_index):
        # Xác định extension
        ext = get_image_extension(url)
        filename = output_dir / f"{index}{ext}"

        # Kiểm tra file đã tồn tại
        if skip_existing and filename.exists():
            print(f"[{index}/{len(urls)}] ⏭️  Bỏ qua {filename.name} (đã tồn tại)")
            skipped += 1
            continue

        print(f"[{index}/{len(urls)}] 📥 Đang tải {filename.name}...")
        print(f"     URL: {url[:80]}...")

        if download_image(url, filename, headers):
            print(f"     ✅ Đã lưu: {filename}")
            downloaded += 1
        else:
            print(f"     ❌ Không thể tải {filename.name}")
            failed += 1

        # Delay giữa các request
        if index < len(urls):
            import time
            time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"✅ Hoàn tất!")
    print(f"   - Đã tải: {downloaded} ảnh")
    print(f"   - Đã bỏ qua: {skipped} ảnh (đã tồn tại)")
    print(f"   - Lỗi: {failed} ảnh")
    print(f"   - Tổng: {len(urls)} URL")
    print(f"{'='*60}")


def main():
    """Hàm main - có thể chỉnh sửa các tham số ở đây"""
    import sys

    # Có thể nhận CSV path từ command line argument
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        # Mặc định dùng file CSV mới nhất trong thư mục csv
        csv_files = list(CSV_DIR.glob("*.csv"))
        if not csv_files:
            print(f"Không tìm thấy file CSV trong {CSV_DIR}")
            print("Cú pháp: python download_images.py [path_to_csv] [output_dir]")
            return

        # Dùng file CSV mới nhất
        csv_path = max(csv_files, key=os.path.getctime)
        print(f"Tự động chọn file CSV mới nhất: {csv_path.name}")

    # Có thể chỉ định output directory
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        # Tạo tên thư mục dựa trên tên CSV file
        output_dir = IMAGES_DIR / csv_path.stem

    # Headers cho iStockPhoto (nếu cần)
    headers = None
    if "istockphoto" in csv_path.stem.lower():
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.istockphoto.com/",
        }

    download_images_from_csv(csv_path, output_dir, headers=headers)


if __name__ == "__main__":
    main()


