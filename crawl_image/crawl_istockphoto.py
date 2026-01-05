import csv
import json
import re
import time
from pathlib import Path
from typing import List, Set

import requests

# Cấu hình
BASE_URL = "https://www.istockphoto.com/vi/search/2/image-film"
PHRASE = "chả giò"
CSV_DIR = Path(__file__).parent / "csv"
CSV_FILE = CSV_DIR / "istockphoto_images.csv"
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 1  # Giây
MAX_PAGES = 100

# Headers để giả lập trình duyệt
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.istockphoto.com/",
    "Origin": "https://www.istockphoto.com",
}


def extract_thumb_urls_from_response(response_text: str) -> List[str]:
    """Trích xuất các thumbUrl từ response"""
    thumb_urls = []

    # Tìm tất cả các thumbUrl trong response
    # Pattern: "thumbUrl":"URL" hoặc 'thumbUrl':'URL'
    patterns = [
        r'"thumbUrl"\s*:\s*"([^"]+)"',  # Double quotes
        r"'thumbUrl'\s*:\s*'([^']+)'",  # Single quotes
        r'"thumbUrl"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',  # With escaped quotes
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        for match in matches:
            try:
                # Decode Unicode escape sequences
                url = match.encode("utf-8").decode("unicode_escape")
            except:
                url = match

            # Loại bỏ các ký tự escape không cần thiết
            url = url.replace("\\/", "/").replace("\\u0026", "&")
            url = url.replace('\\"', '"').replace("\\'", "'")

            # Chỉ thêm URL hợp lệ
            if url.startswith("http") and url not in thumb_urls:
                thumb_urls.append(url)

    return thumb_urls


def load_existing_urls(csv_path: Path) -> Set[str]:
    """Load các URL đã có từ CSV"""
    if not csv_path.exists():
        return set()

    urls = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[0]:
                    urls.add(row[0].strip())
    except Exception as e:
        print(f"Lỗi khi đọc {csv_path}: {e}")
    return urls


def append_urls_to_csv(csv_path: Path, urls: List[str]) -> None:
    """Thêm các URL mới vào CSV"""
    if not urls:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["url"])
        for url in urls:
            writer.writerow([url])


def get_images_from_page(phrase: str, page: int) -> List[str]:
    """Lấy danh sách URL ảnh từ một trang"""
    # URL với page parameter
    url = f"{BASE_URL}?phrase={phrase}&page={page}"

    try:
        print(f"📄 Đang tải trang {page}...")
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Tìm thumbUrl trong response text
        thumb_urls = extract_thumb_urls_from_response(response.text)

        # Nếu không tìm thấy bằng regex, thử tìm trong JSON nếu có
        if not thumb_urls:
            try:
                data = response.json()
                json_str = json.dumps(data)
                thumb_urls = extract_thumb_urls_from_response(json_str)
            except (json.JSONDecodeError, ValueError):
                pass

        return thumb_urls

    except Exception as e:
        print(f"  ❌ Lỗi khi lấy trang {page}: {e}")
        return []


def crawl_istockphoto(
    phrase: str = PHRASE,
    csv_file: Path = CSV_FILE,
    max_pages: int = MAX_PAGES,
    delay: float = DELAY_BETWEEN_REQUESTS,
) -> None:
    """Crawl URL ảnh từ iStockPhoto và lưu vào CSV"""
    seen_urls = load_existing_urls(csv_file)
    print(f"Đã load {len(seen_urls)} URL từ file có sẵn.")

    total_new = 0

    # Lặp qua các trang
    for page in range(1, max_pages + 1):
        print(f"\n{'='*60}")
        print(f"Trang {page}/{max_pages}")
        print(f"{'='*60}")

        # Lấy danh sách URL ảnh từ trang này
        thumb_urls = get_images_from_page(phrase, page)

        if not thumb_urls:
            print(f"  ⚠️  Không tìm thấy ảnh nào ở trang {page}")
            time.sleep(delay)
            continue

        print(f"  ✅ Tìm thấy {len(thumb_urls)} ảnh")

        # Lọc các URL mới
        new_urls = [url for url in thumb_urls if url not in seen_urls]

        if new_urls:
            append_urls_to_csv(csv_file, new_urls)
            for url in new_urls:
                seen_urls.add(url)
            total_new += len(new_urls)
            print(f"  💾 Thêm {len(new_urls)} URL mới (tổng mới {total_new})")
        else:
            print(f"  ⏭️  Không có URL mới ở trang {page}")

        # Delay giữa các trang
        time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"✅ Hoàn tất! Đã thu thập {total_new} URL mới")
    print(f"Dữ liệu được lưu trong file: {csv_file}")
    print(f"{'='*60}")


def main():
    """Hàm main"""
    crawl_istockphoto()


if __name__ == "__main__":
    main()


