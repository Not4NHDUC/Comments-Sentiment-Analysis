"""
Lazada crawler — uses Playwright headless browser.

Review API: acs-m.lazada.vn/h5/mtop.lazada.review.item.getpcreviewlist
Response structure: data.module.reviews (list), data.paging.totalPages
Product URL format: https://www.lazada.vn/products/pdp-i{item_id}.html
"""

import asyncio
import re
import json
from crawlers.utils import auto_label, save_reviews

try:
    from playwright.async_api import async_playwright, Page, Response
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[lazada] playwright not installed")

SEARCH_TERMS = [
    "samsung dien thoai", "laptop gaming", "tai nghe bluetooth", "kem duong da",
    "sua rua mat", "ao thun nam", "giay the thao", "noi chien khong dau",
    "binh nuoc giu nhiet", "may say toc", "dong ho thong minh", "son moi",
    "gau bong do choi", "may tinh bang", "sach day nau an",
    "may xay sinh to", "balo tui xach", "dau goi dau", "nuoc hoa",
    "dien thoai iphone", "may tinh xach tay", "man hinh may tinh",
    "tu lanh samsung", "may giat lg", "dieu hoa panasonic",
    "chan ga goi", "boc an nhua", "ca phe hop", "sua bot cho be",
    "mi an lien", "vit loc nuoc", "den ngu led", "quat dieu hoa",
    "bep tu sharp", "lo vi song", "noi com dien",
    "kinh mat thoi trang", "dong ho nu", "nhan cuoi",
    "giay sandal nu", "tui vi nam", "thot non ao dai",
]

PRODUCT_URL_RE = re.compile(r"pdp-i(\d+)\.html", re.IGNORECASE)
REVIEW_API_RE = re.compile(r"acs-m\.lazada\.vn.+getpcreviewlist", re.IGNORECASE)


async def discover_via_api(page, term: str) -> list[str]:
    """Use in-browser fetch() to call Lazada's search API with session cookies."""
    try:
        js = f"""
        async () => {{
            try {{
                const resp = await fetch('/catalog/?ajax=true&q={term.replace("'","").replace('"','').replace(' ', '+')}', {{
                    headers: {{ 'x-requested-with': 'XMLHttpRequest', 'accept': 'application/json' }}
                }});
                if(!resp.ok) return null;
                const text = await resp.text();
                return text.substring(0, 50000);
            }} catch(e) {{ return null; }}
        }}
        """
        result = await page.evaluate(js)
        if result:
            return PRODUCT_URL_RE.findall(result)
    except Exception:
        pass
    return []


async def discover_products(context, max_total: int = 200) -> list[tuple[str, str]]:
    """Return list of (item_id, product_name) pairs discovered via search."""
    found: dict[str, str] = {}
    page = await context.new_page()

    for idx, term in enumerate(SEARCH_TERMS):
        if len(found) >= max_total:
            break
        try:
            # Every 3 searches, go back to homepage to refresh anti-bot trust
            if idx % 3 == 0:
                await page.goto("https://www.lazada.vn/", wait_until="domcontentloaded", timeout=25000)
                await page.wait_for_timeout(1500)

            url = f"https://www.lazada.vn/catalog/?q={term.replace(' ', '+')}"
            await page.goto(url, wait_until="domcontentloaded", timeout=35000)
            await page.wait_for_timeout(3500)
            await page.evaluate("window.scrollTo(0, 1500)")
            await page.wait_for_timeout(1500)

            # Try both DOM links and in-browser API
            links_dom = await page.evaluate("""
                () => [...new Set(
                    Array.from(document.querySelectorAll('a[href*="/products/"]'))
                        .map(a => a.href)
                )]
            """)
            ids_from_dom = [PRODUCT_URL_RE.search(l).group(1) for l in links_dom if PRODUCT_URL_RE.search(l)]
            ids_from_api = await discover_via_api(page, term)
            all_ids = list(dict.fromkeys(ids_from_dom + ids_from_api))  # deduplicate preserving order

            added = 0
            for item_id in all_ids:
                if item_id not in found:
                    found[item_id] = term[:40]
                    added += 1

            print(f"  [lazada discover] '{term}': +{added} products (total {len(found)})")
        except Exception as e:
            print(f"  [lazada discover] Error on '{term}': {e}")
        await asyncio.sleep(1.5)

    await page.close()
    return list(found.items())


def _parse_reviews(body: dict, item_id: str, product_name: str) -> list[dict]:
    reviews = []
    try:
        items = body["data"]["module"]["reviews"]
    except (KeyError, TypeError):
        return reviews

    for item in items:
        rating = int(item.get("rating", 0) or 0)

        # Join all content parts into one text
        parts = item.get("reviewContentList") or []
        texts = []
        for p in parts:
            content = (p.get("content") or "").strip()
            attr = (p.get("attribute") or "").strip()
            if attr and content:
                texts.append(f"{attr}: {content}")
            elif content:
                texts.append(content)
        review_text = " | ".join(texts)

        date_str = item.get("reviewTime", "")

        reviews.append({
            "platform": "lazada",
            "product_id": item_id,
            "product_name": product_name,
            "rating": rating,
            "review_text": review_text,
            "date": date_str,
            "label": auto_label(rating),
        })

    return reviews


async def crawl_product(page: Page, item_id: str, product_name: str) -> list[dict]:
    reviews = []
    captured: list[dict] = []

    async def on_response(response: Response):
        if REVIEW_API_RE.search(response.url):
            try:
                body = await response.json()
                captured.append(body)
            except Exception:
                pass

    page.on("response", on_response)

    url = f"https://www.lazada.vn/products/pdp-i{item_id}.html"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=40000)
        await page.wait_for_timeout(3000)

        # Scroll to reviews section
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.65)")
        await page.wait_for_timeout(2000)

        # Try clicking the reviews tab
        for selector in [
            "text=Đánh giá", "[data-aplus-ae='customer_reviews']",
            "[href='#reviews']", ".pdp-block__title-reviews",
        ]:
            try:
                el = page.locator(selector).first
                if await el.is_visible(timeout=1500):
                    await el.scroll_into_view_if_needed()
                    await el.click()
                    await page.wait_for_timeout(1500)
                    break
            except Exception:
                pass

        # Process first page of captured reviews
        for body in captured:
            reviews.extend(_parse_reviews(body, item_id, product_name))

        # Determine total pages from the first response
        total_pages = 1
        if captured:
            try:
                total_pages = int(captured[0]["data"]["paging"]["totalPages"])
            except (KeyError, TypeError, ValueError):
                pass

        print(f"  [lazada] item {item_id}: total_pages={total_pages}, got {len(reviews)} so far")

        # Save page HTML once for pagination debugging
        import os
        debug_html = "lazada_product_debug.html"
        if len(reviews) > 0 and not os.path.exists(debug_html):
            try:
                content = await page.content()
                with open(debug_html, "w", encoding="utf-8") as _f:
                    _f.write(content)
            except Exception:
                pass

        # Paginate: click "Next page" up to max 15 pages
        for pg in range(2, min(total_pages + 1, 16)):
            captured.clear()
            try:
                next_btn = page.locator(
                    "li.ant-pagination-next:not(.ant-pagination-disabled) button, "
                    "button.ant-pagination-next:not([disabled]), "
                    ".next-pagination-item-next:not(.disabled)"
                ).first
                if await next_btn.count() == 0:
                    break
                if await next_btn.is_disabled(timeout=1500):
                    break
                await next_btn.scroll_into_view_if_needed()
                await next_btn.click()
                await page.wait_for_timeout(2000)
                for body in captured:
                    batch = _parse_reviews(body, item_id, product_name)
                    reviews.extend(batch)
                    print(f"    page {pg}/{total_pages}: +{len(batch)}")
            except Exception as e:
                print(f"    pagination stopped at page {pg}: {e}")
                break

    except Exception as e:
        print(f"  [lazada] Navigation error for {item_id}: {e}")
    finally:
        page.remove_listener("response", on_response)

    return reviews


async def crawl_all(target: int = 800) -> list[dict]:
    if not PLAYWRIGHT_AVAILABLE:
        print("[lazada] Playwright not available, skipping.")
        return []

    all_reviews: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="vi-VN",
            timezone_id="Asia/Ho_Chi_Minh",
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)

        # Warm up
        warmup = await context.new_page()
        try:
            await warmup.goto("https://www.lazada.vn/", wait_until="domcontentloaded", timeout=30000)
            await warmup.wait_for_timeout(2000)
        except Exception:
            pass
        await warmup.close()

        # Discover real product IDs
        print("[lazada] Discovering products from search results...")
        products = await discover_products(context)
        print(f"[lazada] Discovered {len(products)} products")

        if not products:
            print("[lazada] No products discovered, aborting.")
            await browser.close()
            return []

        # Crawl reviews
        page = await context.new_page()
        for item_id, product_name in products:
            if len(all_reviews) >= target:
                break
            print(f"[lazada] Crawling item {item_id} ({len(all_reviews)}/{target})")
            try:
                batch = await crawl_product(page, item_id, product_name)
                all_reviews.extend(batch)
                print(f"  [lazada] Got {len(batch)} reviews, total now {len(all_reviews)}")
            except Exception as e:
                print(f"  [lazada] Error: {e}")
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

    print(f"[lazada] Total unique reviews: {len(unique)}")
    save_reviews(unique, "lazada", append=True)
    return unique


def run(target: int = 800) -> list[dict]:
    return asyncio.run(crawl_all(target))


if __name__ == "__main__":
    run()
