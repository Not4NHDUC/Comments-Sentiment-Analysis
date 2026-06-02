"""
Shopee crawler — uses Playwright headless browser.

Product discovery: after warming up with homepage, use in-browser fetch()
to call Shopee's internal search API (same session cookies, bypasses rendering).
Review API: shopee.vn/api/v2/item/get_ratings
"""

import asyncio
import datetime
import json
import re
from crawlers.utils import auto_label, save_reviews

try:
    from playwright.async_api import async_playwright, Page, Response
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[shopee] playwright not installed - run: pip install playwright && playwright install chromium")

try:
    from playwright_stealth import Stealth as _Stealth
    _STEALTH_AVAILABLE = True
except ImportError:
    _STEALTH_AVAILABLE = False

SEARCH_KEYWORDS = [
    "samsung dien thoai", "laptop gaming", "tai nghe bluetooth",
    "kem duong da", "sua rua mat", "ao thun nam", "giay the thao",
    "sach ky nang", "mi tom", "bim pampers", "nuoc hoa",
    "dong ho nam", "may xay sinh to", "gau bong thu nhoi bong",
    "son moi mau dep", "serum vitamin c", "dien thoai iphone",
    "balo du lich", "am dun nuoc",
]

RATING_API_RE = re.compile(r"shopee\.vn/api/v2/item/get_ratings")


async def discover_via_api(page: Page, keyword: str, limit: int = 40) -> list[tuple[str, str]]:
    """Use in-browser fetch() to call Shopee search API — uses real browser session/cookies."""
    results = []
    try:
        js = f"""
        async () => {{
            try {{
                const resp = await fetch('/api/v4/search/search_items?by=relevancy&keyword={keyword.replace("'","").replace('"','')}&limit={limit}&newest=0&order=desc&page_type=search&version=2', {{
                    headers: {{
                        'x-api-source': 'pc',
                        'x-requested-with': 'XMLHttpRequest',
                        'accept': 'application/json',
                    }}
                }});
                if(!resp.ok) return null;
                const data = await resp.json();
                return data;
            }} catch(e) {{
                return {{'error': e.toString()}};
            }}
        }}
        """
        data = await page.evaluate(js)
        if not data or "error" in data:
            return results

        items = (data.get("data") or {}).get("item_basic_list", []) or \
                data.get("items", []) or []

        for item in items:
            shop_id = str(item.get("shopid", ""))
            item_id = str(item.get("itemid", ""))
            if shop_id and item_id:
                results.append((shop_id, item_id))
    except Exception as e:
        print(f"  [shopee discover api] Error for '{keyword}': {e}")
    return results


async def discover_products(context, max_total: int = 120) -> list[tuple[str, str, str]]:
    """Discover products using in-browser fetch API calls."""
    found: dict[str, tuple[str, str, str]] = {}
    page = await context.new_page()

    # Warm up on homepage first to get session cookies
    try:
        await page.goto("https://shopee.vn/", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)
    except Exception as e:
        print(f"  [shopee discover] Warmup error: {e}")

    for keyword in SEARCH_KEYWORDS:
        if len(found) >= max_total:
            break
        products = await discover_via_api(page, keyword)
        added = 0
        for shop_id, item_id in products:
            key = f"{shop_id}_{item_id}"
            if key not in found:
                found[key] = (shop_id, item_id, keyword)
                added += 1
        print(f"  [shopee discover] '{keyword}': +{added} products (total {len(found)})")
        await asyncio.sleep(1)

    await page.close()
    return list(found.values())


async def crawl_product(page: Page, shop_id: str, item_id: str, product_name: str) -> list[dict]:
    reviews = []
    captured: list[dict] = []

    async def on_response(response: Response):
        if RATING_API_RE.search(response.url):
            try:
                body = await response.json()
                captured.append(body)
            except Exception:
                pass

    page.on("response", on_response)

    url = f"https://shopee.vn/product/{shop_id}/{item_id}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=35000)
        await page.wait_for_timeout(3500)

        # Scroll to trigger review loading
        for pct in [0.4, 0.6, 0.75, 0.9]:
            await page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {pct})")
            await page.wait_for_timeout(700)
        await page.wait_for_timeout(1000)

        def parse_batch(bodies):
            batch = []
            for body in bodies:
                items = (body.get("data") or {}).get("ratings", [])
                for item in items:
                    rating = int(item.get("rating_star", 0))
                    text = (item.get("comment") or "").strip()
                    ctime = item.get("ctime", 0)
                    date_str = ""
                    if ctime:
                        date_str = datetime.datetime.fromtimestamp(ctime, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
                    batch.append({
                        "platform": "shopee",
                        "product_id": item_id,
                        "product_name": product_name,
                        "rating": rating,
                        "review_text": text,
                        "date": date_str,
                        "label": auto_label(rating),
                    })
            return batch

        reviews.extend(parse_batch(captured))

        # Paginate via next button clicks
        prev_count = len(reviews)
        for _pg in range(15):
            captured.clear()
            try:
                next_btn = page.locator(
                    "button[aria-label='Next page'], "
                    ".shopee-icon-button--right, "
                    "button.shopee-page-controller__next-btn, "
                    "[data-sqe='btn-next']"
                ).first
                if await next_btn.count() == 0:
                    break
                if await next_btn.is_disabled(timeout=1000):
                    break
                await next_btn.click()
                await page.wait_for_timeout(2000)
                reviews.extend(parse_batch(captured))
                if len(reviews) == prev_count:
                    break
                prev_count = len(reviews)
            except Exception:
                break

    except Exception as e:
        print(f"  [shopee] Error on {item_id}: {e}")
    finally:
        page.remove_listener("response", on_response)

    return reviews


async def crawl_all(target: int = 800) -> list[dict]:
    if not PLAYWRIGHT_AVAILABLE:
        print("[shopee] Playwright not available, skipping.")
        return []

    all_reviews: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled", "--disable-infobars"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="vi-VN",
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)

        # Apply stealth patches if available
        if _STEALTH_AVAILABLE:
            _stealth_instance = _Stealth()

        # Discover products via in-browser API
        print("[shopee] Discovering products via in-browser search API...")
        products = await discover_products(context)
        print(f"[shopee] Discovered {len(products)} products")

        if not products:
            print("[shopee] No products discovered, aborting.")
            await browser.close()
            return []

        page = await context.new_page()
        for shop_id, item_id, keyword in products:
            if len(all_reviews) >= target:
                break
            print(f"[shopee] Crawling item {item_id} ({len(all_reviews)}/{target})")
            try:
                batch = await crawl_product(page, shop_id, item_id, keyword)
                all_reviews.extend(batch)
                print(f"  [shopee] Got {len(batch)}, total: {len(all_reviews)}")
            except Exception as e:
                print(f"  [shopee] Error: {e}")
            await asyncio.sleep(2)

        await browser.close()

    # Deduplicate
    seen: set = set()
    unique = []
    for r in all_reviews:
        key = (r["product_id"], r["review_text"][:80])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"[shopee] Total unique reviews: {len(unique)}")
    save_reviews(unique, "shopee", append=True)
    return unique


def run(target: int = 800) -> list[dict]:
    return asyncio.run(crawl_all(target))


if __name__ == "__main__":
    run()
