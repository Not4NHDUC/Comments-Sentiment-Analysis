import csv
import io
import os
import random

import requests

from crawlers.utils import save_reviews, REVIEW_FIELDS

# Sentiment class order from the original dataset script: 0=negative, 1=neutral, 2=positive
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Direct Google Drive download links (sentences + sentiments for train split)
# Note: space removed from the original sentiments URL typo
_TRAIN_SENTENCES_URL = "https://drive.google.com/uc?id=1nzak5OkrheRV1ltOGCXkT671bmjODLhP&export=download"
_TRAIN_SENTIMENTS_URL = "https://drive.google.com/uc?id=1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv&export=download"


def _gdrive_download(url: str) -> str:
    """Download a small Google Drive file and return its text content."""
    session = requests.Session()
    resp = session.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    # Google Drive adds a virus-scan warning for larger files; follow the confirm link if present
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        # Extract the confirm token and retry
        token = None
        for chunk in resp.iter_content(chunk_size=32768):
            text = chunk.decode("utf-8", errors="replace")
            import re
            m = re.search(r'confirm=([0-9A-Za-z_\-]+)', text)
            if m:
                token = m.group(1)
                break
        if token:
            confirm_url = url + f"&confirm={token}"
            resp = session.get(confirm_url, stream=True, timeout=30)
            resp.raise_for_status()

    return resp.content.decode("utf-8")


def _load_raw() -> list[dict]:
    """Download train sentences + sentiments and return list of dicts."""
    print("[vsfc] Downloading UIT-VSFC train sentences...")
    sentences_text = _gdrive_download(_TRAIN_SENTENCES_URL)
    print("[vsfc] Downloading UIT-VSFC train sentiments...")
    sentiments_text = _gdrive_download(_TRAIN_SENTIMENTS_URL)

    sentences = [s.strip() for s in sentences_text.splitlines() if s.strip()]
    sentiments = [s.strip() for s in sentiments_text.splitlines() if s.strip()]

    rows = []
    for sentence, sentiment_str in zip(sentences, sentiments):
        try:
            label = LABEL_MAP[int(sentiment_str)]
        except (ValueError, KeyError):
            continue
        rows.append({
            "platform": "uit_vsfc",
            "product_id": "vsfc",
            "product_name": "student_feedback",
            "rating": 0,
            "review_text": sentence,
            "date": "",
            "label": label,
        })
    return rows


def load_vsfc(target_per_class: int = 400) -> list[dict]:
    all_rows = _load_raw()

    buckets: dict[str, list[dict]] = {"Negative": [], "Neutral": [], "Positive": []}
    for row in all_rows:
        label = row["label"]
        if label in buckets:
            buckets[label].append(row)

    result: list[dict] = []
    for label, rows in buckets.items():
        result.extend(random.sample(rows, min(target_per_class, len(rows))))

    return result


def supplement_data(existing_csv: str, target_per_class: int = 400) -> list[dict]:
    existing: list[dict] = []
    if os.path.exists(existing_csv):
        with open(existing_csv, encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))

    counts: dict[str, int] = {"Negative": 0, "Neutral": 0, "Positive": 0}
    for r in existing:
        label = r.get("label", "")
        if label in counts:
            counts[label] += 1

    gaps = {label: max(0, target_per_class - counts[label]) for label in counts}

    if all(g == 0 for g in gaps.values()):
        print("[vsfc] All classes already at target, no supplement needed.")
        return existing

    all_vsfc = _load_raw()

    buckets: dict[str, list[dict]] = {"Negative": [], "Neutral": [], "Positive": []}
    for row in all_vsfc:
        label = row["label"]
        if label in buckets:
            buckets[label].append(row)

    added: dict[str, int] = {"Negative": 0, "Neutral": 0, "Positive": 0}
    new_rows: list[dict] = []
    for label, gap in gaps.items():
        if gap == 0:
            continue
        sample = random.sample(buckets[label], min(gap, len(buckets[label])))
        new_rows.extend(sample)
        added[label] += len(sample)

    print(
        f"[vsfc] Supplemented {added['Negative']} Negative, "
        f"{added['Neutral']} Neutral, {added['Positive']} Positive from UIT-VSFC"
    )

    merged = existing + new_rows
    out_dir = os.path.dirname(existing_csv) or "data/raw"
    save_reviews(merged, platform="combined", out_dir=out_dir, append=True)
    return merged
