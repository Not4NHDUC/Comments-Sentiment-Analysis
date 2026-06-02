"""
Lazada recovery crawl — runs with the exact approach that worked the first time.
Uses very conservative pacing to avoid bot detection.
Run this after waiting at least 30-60 minutes from the last Lazada crawl attempt.
"""

import asyncio
import re
import json
from crawlers.utils import auto_label, save_reviews

try:
    from playwright.async_api import async_playwright, Response
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Top products from first successful run (highest review counts)
TOP_ITEMS = [
    "3038753147",  # 214 pages
    "3013274260",  # 71 pages
    "2829978834",  # 20 pages
    "2949443299",  # 28 pages
    "3332640654",  # 30 pages
    "2137748664",  "2974655685", "3082255791", "2890874224", "3229045398",
    "2890686126",  "3010182283", "3213776114", "2948636120", "2829959779",
    "2018163572",  "3213682260", "2948663192", "2890513328", "2890356191",
    "2948335940",  "2948802023", "3008309320", "2154532236", "2437150867",
    "2260546813",  "2890882018", "2948525081", "2260606871", "3008210780",
    "2406123599",  "3010119056", "2949443299", "3229148138", "3305235653",
    "2658585204",  "13387874348",
]

REVIEW_API_RE = re.compile(r"acs-m\.lazada\.vn.+getpcreviewlist", re.IGNORECASE)


def _parse_reviews(body, item_id, product_name):
    reviews = []
    try:
        items = body["data"]["module"]["reviews"]
    except (KeyError, TypeError):
        return reviews
    for item in items:
        rating = int(item.get("rating", 0) or 0)
        parts = item.get("reviewContentList") or []
        texts = []
        for p in parts:
            content = (p.get("content") or "").strip()
            attr = (p.get("attribute") or "").strip()
            if attr and content:
                texts.append(f"{attr}: {content}")
            elif content:
                texts.append(content)
        reviews.append({
            "platform": "lazada",
            "product_id": item_id,
            "product_name": product_name,
            "rating": rating,
            "review_text": " | ".join(texts),
            "date": item.get("reviewTime", ""),
            "label": auto_label(rating),
        })
    return reviews


async def crawl_item(page, item_id):
    reviews = []
    captured = []

    async def on_resp(r):
        if REVIEW_API_RE.search(r.url):
            try:
                captured.append(await r.json())
            except Exception:
                pass

    page.on("response", on_resp)
    try:
        await page.goto(f"https://www.lazada.vn/products/pdp-i{item_id}.html",
                        wait_until="domcontentloaded", timeout=40000)
        await page.wait_for_timeout(4000)
        for pct in [0.3, 0.5, 0.65, 0.8]:
            await page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {pct})")
            await page.wait_for_timeout(1200)

        for body in captured:
            reviews.extend(_parse_reviews(body, item_id, f"item_{item_id}"))

        total_pages = 1
        if captured:
            try:
                total_pages = int(captured[0]["data"]["paging"]["totalPages"])
            except Exception:
                pass

        print(f"  item {item_id}: pages={total_pages}, first_page={len(reviews)}")
    except Exception as e:
        print(f"  item {item_id} error: {e}")
    finally:
        page.remove_listener("response", on_resp)
    return reviews


async def recover(target: int = 800):
    if not PLAYWRIGHT_AVAILABLE:
        return []

    all_reviews = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="vi-VN",
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)

        page = await context.new_page()
        # Homepage warmup
        await page.goto("https://www.lazada.vn/", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)
        # Search page visit to establish trust
        await page.goto("https://www.lazada.vn/catalog/?q=samsung+dien+thoai",
                        wait_until="domcontentloaded", timeout=35000)
        await page.wait_for_timeout(4000)
        await page.evaluate("window.scrollTo(0, 1500)")
        await page.wait_for_timeout(2000)

        for item_id in TOP_ITEMS:
            if len(all_reviews) >= target:
                break
            print(f"[lazada recovery] item {item_id} ({len(all_reviews)}/{target})")
            batch = await crawl_item(page, item_id)
            all_reviews.extend(batch)
            await asyncio.sleep(3)  # Conservative 3s delay

        await browser.close()

    seen = set()
    unique = []
    for r in all_reviews:
        key = (r["product_id"], r["review_text"][:80])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"[lazada recovery] Got {len(unique)} reviews")
    if unique:
        save_reviews(unique, "lazada", append=True)
    return unique


if __name__ == "__main__":
    asyncio.run(recover())
