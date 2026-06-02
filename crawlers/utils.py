import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

REVIEW_FIELDS = ["platform", "product_id", "product_name", "rating", "review_text", "date", "label"]


def get_random_ua():
    return random.choice(USER_AGENTS)


def auto_label(rating: int) -> str:
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"


def save_reviews(reviews: list[dict], platform: str, out_dir: str = "data/raw", append: bool = False):
    """
    Save reviews to CSV + JSON.
    If append=True, merge with existing data and deduplicate.
    Never overwrites existing data with fewer records.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{platform}.csv")
    json_path = os.path.join(out_dir, f"{platform}.json")

    existing: list[dict] = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, encoding="utf-8-sig") as f:
                existing = list(csv.DictReader(f))
        except Exception:
            pass

    # Drop rows with no review text before any merging or saving
    reviews = [r for r in reviews if r.get("review_text", "").strip()]

    if append and existing:
        # Merge: deduplicate by (product_id, review_text prefix)
        seen = {(r.get("product_id", ""), r.get("review_text", "")[:80]) for r in existing}
        for r in reviews:
            key = (r.get("product_id", ""), r.get("review_text", "")[:80])
            if key not in seen:
                existing.append(r)
                seen.add(key)
        reviews = existing

    # Safety: never overwrite good data with fewer records
    if len(reviews) < len(existing):
        print(f"[{platform}] WARNING: new data ({len(reviews)}) < existing ({len(existing)}), keeping existing")
        return csv_path

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(reviews)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    print(f"[{platform}] Saved {len(reviews)} reviews -> {csv_path}")
    return csv_path


def merge_all(out_dir: str = "data/raw"):
    combined = []
    for platform in ["tiki", "shopee", "lazada"]:
        path = os.path.join(out_dir, f"{platform}.csv")
        if not os.path.exists(path):
            print(f"[merge] Missing {path}, skipping.")
            continue
        with open(path, encoding="utf-8-sig") as f:
            combined.extend(list(csv.DictReader(f)))

    out_path = os.path.join(out_dir, "combined.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(combined)

    print(f"[merge] Combined {len(combined)} total reviews -> {out_path}")
    return out_path
