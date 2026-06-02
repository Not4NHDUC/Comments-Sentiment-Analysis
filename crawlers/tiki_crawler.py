"""
Tiki crawler — uses public REST API: GET /api/v2/reviews?product_id=&page=&limit=20
No browser needed; plain requests with polite rate-limiting.
"""

import time
import requests
from crawlers.utils import auto_label, save_reviews, get_random_ua

API_URL = "https://tiki.vn/api/v2/reviews"
PRODUCT_API = "https://tiki.vn/api/v2/products/{product_id}"

# Popular Tiki product IDs with many reviews — discovered via category browsing
# Sorted by review count descending for fastest data collection
PRODUCT_IDS = [
    "74021317",  "52789367",  "273819808", "7833728",   "123836523",
    "59143353",  "356188",    "32951822",  "35761444",  "157240665",
    "273598230", "76013378",  "912679",    "54266567",  "59504295",
    "1672157",   "10240037",  "632680",    "1080002",   "133696713",
    "5560235",   "1672153",   "7982628",   "767101",    "98565212",
    "359484",    "631034",    "15973974",  "1935009",   "70816541",
    "91947689",  "271380890", "50322710",  "32951828",  "170708233",
    "54058501",  "13920941",  "997535",    "22448892",  "24028050",
    "13446508",  "56609155",  "176899932", "54665",     "2042227",
    "112289039", "80857582",  "133683792", "54058503",  "540040",
    "277995357", "277728224", "176250783", "176251118", "273201159",
    "273717706", "275220818", "275220816", "273472564", "277619147",
    # Extended list for 1000+ review target
    "109017985", "3304875",   "72202103",  "10005245",  "3954355",
    "17336364",  "276833162", "4780917",   "57325187",  "13419678",
    "7369223",   "2738475",   "260844985", "276451290", "8886007",
    "10581200",  "58811363",  "685882",    "56859344",  "274032614",
    "45663008",  "540039",    "48362502",  "278004511", "140416370",
    "631496",    "133682930", "2454357",   "98329162",  "68841415",
    "270831162", "276048594", "46022902",  "272965028", "71073459",
    "3454049",   "36609163",  "31710791",  "28694099",  "39985599",
]


def fetch_product_name(product_id: str, session: requests.Session) -> str:
    try:
        r = session.get(
            PRODUCT_API.format(product_id=product_id),
            headers={"User-Agent": get_random_ua()},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("name", f"product_{product_id}")
    except Exception:
        pass
    return f"product_{product_id}"


def crawl_product(product_id: str, session: requests.Session, max_pages: int = 25) -> list[dict]:
    reviews = []
    product_name = fetch_product_name(product_id, session)
    time.sleep(0.3)

    for page in range(1, max_pages + 1):
        try:
            r = session.get(
                API_URL,
                params={"product_id": product_id, "page": page, "limit": 20},
                headers={
                    "User-Agent": get_random_ua(),
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
                    "Referer": f"https://tiki.vn/p{product_id}",
                },
                timeout=15,
            )
        except requests.RequestException as e:
            print(f"  [tiki] Request error page {page}: {e}")
            break

        if r.status_code != 200:
            print(f"  [tiki] HTTP {r.status_code} on product {product_id} page {page}")
            break

        data = r.json()
        items = data.get("data", [])
        if not items:
            break

        for item in items:
            rating = int(item.get("rating", 0))
            review_text = (item.get("content") or "").strip()
            date_str = item.get("created_at", "")
            if isinstance(date_str, int):
                import datetime as _dt
                date_str = _dt.datetime.fromtimestamp(date_str, tz=_dt.timezone.utc).strftime("%Y-%m-%d")

            reviews.append({
                "platform": "tiki",
                "product_id": product_id,
                "product_name": product_name,
                "rating": rating,
                "review_text": review_text,
                "date": date_str,
                "label": auto_label(rating),
            })

        paging = data.get("paging", {})
        total_pages = paging.get("last_page", 1)
        print(f"  [tiki] product {product_id} page {page}/{total_pages} -> {len(items)} reviews")

        if page >= total_pages:
            break

        time.sleep(0.5)

    return reviews


def count_labels(reviews: list[dict]) -> dict:
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in reviews:
        label = r.get("label", "")
        if label in counts:
            counts[label] += 1
    return counts


def run(target: int = 1200) -> list[dict]:
    session = requests.Session()
    all_reviews: list[dict] = []

    # Skip products already in the existing CSV (append mode)
    import csv as _csv, os as _os
    existing_pids: set = set()
    existing_path = "data/raw/tiki.csv"
    if _os.path.exists(existing_path):
        try:
            with open(existing_path, encoding="utf-8-sig") as _f:
                for row in _csv.DictReader(_f):
                    existing_pids.add(row.get("product_id", ""))
        except Exception:
            pass

    for pid in PRODUCT_IDS:
        if len(all_reviews) >= target:
            break
        if pid in existing_pids:
            continue  # Already crawled
        print(f"[tiki] Crawling product {pid} ({len(all_reviews)}/{target} so far)")
        reviews = crawl_product(pid, session)
        all_reviews.extend(reviews)
        time.sleep(1)

    # Second pass: top up under-represented labels
    counts = count_labels(all_reviews)
    if counts["Negative"] < 300 or counts["Neutral"] < 200:
        print(f"[tiki] Imbalanced: {counts}, starting second pass for low-rating reviews...")
        for pid in PRODUCT_IDS:
            if counts["Negative"] >= 300 and counts["Neutral"] >= 200:
                break
            reviews = crawl_product(pid, session, max_pages=25)
            low_reviews = [r for r in reviews if int(r["rating"]) <= 3]
            all_reviews.extend(low_reviews)
            counts = count_labels(all_reviews)
            time.sleep(1)

    # Deduplicate by (product_id, review_text)
    seen = set()
    unique = []
    for r in all_reviews:
        key = (r["product_id"], r["review_text"][:80])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"[tiki] Total unique reviews: {len(unique)}")
    save_reviews(unique, "tiki", append=True)
    return unique


if __name__ == "__main__":
    run()
