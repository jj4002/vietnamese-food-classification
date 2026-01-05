import asyncio
import csv
import os
from pathlib import Path
from typing import List, Set

from undetected_playwright.async_api import async_playwright

# Cấu hình
SEARCH_URL = "https://www.flickr.com/search/?view_all=1&text=x%C3%B4i+x%C3%A9o"
CSV_DIR = Path(__file__).parent / "csv"
CSV_FILE = CSV_DIR / "flickr_images.csv"
POLL_INTERVAL_SECONDS = 5
MONITOR_DURATION_MINUTES = 5
COOKIE_STRING = (
    "notice_behavior=implied,us; usprivacy=1---; _ga=GA1.1.1543921934.1763886951; "
    "sp=a1bf6e12-34c9-4959-bc8a-bd4bbad83221; _ga_RHPTTDDLCW=GS2.1.s1763886950$o1$g0$t1763887071$j60$l0$h53485459; "
    "AMCVS_48E815355BFE96970A495CD0%40AdobeOrg=1; AMCV_48E815355BFE96970A495CD0%40AdobeOrg=281789898%7CMCMID%7C74729185305597836141326540079632826730%7CMCAAMLH-1764491875%7C3%7CMCAAMB-1764491875%7C6G1ynYcLPuiQxYZrsz_pkqfLG9yMXBpb2zX5dvJdYQJzPXImdj0y%7CMCOPTOUT-1763894275s%7CNONE%7CvVersion%7C4.1.0; "
    "s_ptc=%5B%5BB%5D%5D; s_tp=738; s_ppv=https%253A%2F%2Fidentity.flickr.com%2F%253FpostSignUp%253Dtrue%2C100%2C100%2C738; "
    "s_sq=smugmugincflickrprodudction%3D%2526c.%2526a.%2526activitymap.%2526page%253Dhttps%25253A%25252F%25252Fidentity.flickr.com%25252Fchange-complete%25252Fsign-up%25252F%2526link%253DSign%252520in%2526region%253Dlogin-form%2526.activitymap%2526.a%2526.c; "
    "xb=862572; cookie_session=203831705%3A68eb4f61d5651ffe7bf955b3a614e135; cookie_accid=203831705; "
    "cookie_epass=68eb4f61d5651ffe7bf955b3a614e135; sa=1769071075%3A203853035%40N07%3A2f7a6f98d3ceaea44cc0faad80d0e4b9; "
    "ccc=%7B%22needsConsent%22%3Afalse%2C%22managed%22%3A0%2C%22changed%22%3A0%2C%22info%22%3A%7B%22cookieBlock%22%3A%7B%22level%22%3A0%2C%22blockRan%22%3A0%7D%7D%2C%22freshServerContext%22%3Atrue%7D; "
    "__ssid=05df3a50c703274112da9e4057518ec; localization=en-us%3Bxx%3Bvn; "
    "flrbp=1763887133-ee2a57cca732e7569739b86bf42b90991c7ac0fc; flrgrp=1763887133-0f3e393ef709057ea0ccd41fff723191a8bfc335; "
    "flrbgdrp=1763887133-1e95eaec77f11fef45481ccc4a2712baecbebcfb; flrbgmrp=1763887133-b78aaa7c11f67a243e2604c133887da1fa4c4bc3; "
    "flrbrst=1763887133-cbff73b054f1cc8dca757a8be7d207e31ef4884d; flrtags=1763887133-a3acf29483fb9ea41af2dc86afa98ecc618eb2c7; "
    "flrbfd=1763887133-f8304175ecab3b80466bf1f6f17cb6ef19cfaeae; flrbrp=1763887133-2b47000916a342e1d8f982b54f92d031974243a2; "
    "flrbpap=1763887133-9753433de5b0c04fcd575e228dce39e77da7c51b; liqpw=1528; liqph=689; _sp_ses.df80=*; "
    "__gads=ID=f847642ed1d2b6e8:T=1763887094:RT=1763991317:S=ALNI_MaqXjHQn4jAvJ1_gKlA7JP3lr1d0w; "
    "__eoi=ID=6b58cf09b6c049ae:T=1763887094:RT=1763991317:S=AA-AfjZjJKAG2sKU5Mpl32gCky7u; "
    "_awl=2.1763991351.5-60388dea906322e3a63744ea361c6d0f-6763652d617369612d6561737431-0; "
    "adCounter=1; "
    "vp=616%2C738%2C1.25%2C15%2Csearch-photos-albums-new-view%3A1513%2Csearch-photos-everyone-view%3A616%2Csearch-photos-contacts-view%3A1513%2Csearch-photos-yours-view%3A1513; "
    "_sp_id.df80=537a0350-761b-48d7-b2fd-f02cc7ed3591.1763886950.2.1763991988.1763887212.793168af-57ab-4ca3-8e07-f26a0b22b79c.169351c6-ac17-4e09-a2a5-4e9471b00ce1.e0f8870c-9a19-4c7c-898d-2d449ebbdf85.1763991371296.51"
)


def normalize_flickr_src(src: str) -> str:
    """Chuẩn hóa URL ảnh từ Flickr"""
    if not src:
        return ""
    src = src.strip()
    if src.startswith("//"):
        return f"https:{src}"
    if src.startswith("http://") or src.startswith("https://"):
        return src
    return ""


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
                if row:
                    urls.add(row[0])
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


async def scrape_visible_image_urls(page) -> List[str]:
    """Lấy danh sách URL ảnh hiển thị trên trang"""
    raw_urls = await page.evaluate(
        """() => Array.from(
                document.querySelectorAll('div.photo-list-photo-container img[src]'),
                img => img.getAttribute('src') || ''
            )"""
    )
    cleaned = []
    for src in raw_urls:
        normalized = normalize_flickr_src(src)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def cookies_from_string(cookie_string: str, domain: str = ".flickr.com") -> List[dict]:
    """Chuyển đổi cookie string thành list cookies"""
    cookies = []
    for part in cookie_string.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        cookies.append(
            {
                "name": name,
                "value": value,
                "domain": domain,
                "path": "/",
            }
        )
    return cookies


async def crawl_flickr(
    search_url: str = SEARCH_URL,
    csv_file: Path = CSV_FILE,
    duration_minutes: int = MONITOR_DURATION_MINUTES,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    headless: bool = False,
) -> None:
    """Crawl URL ảnh từ Flickr và lưu vào CSV"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        try:
            cookies = cookies_from_string(COOKIE_STRING)
            await page.context.add_cookies(cookies)
            print(f"Đã nạp {len(cookies)} cookies từ COOKIE_STRING")
        except Exception as e:
            print(f"Lỗi khi nạp cookies (tiếp tục không dùng cookie): {e}")

        print(f"Mở trang: {search_url}")
        await page.goto(search_url, wait_until="domcontentloaded")
        print("Bạn có thể tự scroll để load thêm ảnh. Script sẽ liên tục kiểm tra và lưu URL mới.")

        seen_urls = load_existing_urls(csv_file)
        print(f"Đã load {len(seen_urls)} URL từ file có sẵn.")

        start_time = asyncio.get_event_loop().time()
        total_new = 0

        while True:
            current_urls = await scrape_visible_image_urls(page)
            new_urls = [url for url in current_urls if url not in seen_urls]

            if new_urls:
                append_urls_to_csv(csv_file, new_urls)
                for url in new_urls:
                    seen_urls.add(url)
                total_new += len(new_urls)
                print(f"Thêm {len(new_urls)} URL mới (tổng mới {total_new}).")
            else:
                print("Chưa thấy URL mới, sẽ kiểm tra lại...")

            elapsed_minutes = (asyncio.get_event_loop().time() - start_time) / 60
            if elapsed_minutes >= duration_minutes:
                print(f"Hết {duration_minutes} phút theo dõi, dừng lại.")
                break

            await asyncio.sleep(poll_interval)

        print(f"Tổng số URL mới thu thập trong phiên này: {total_new}")
        print(f"Dữ liệu được lưu trong file: {csv_file}")
        await browser.close()


async def main():
    """Hàm main"""
    await crawl_flickr()


if __name__ == "__main__":
    asyncio.run(main())


