"""
Vietnamese e-commerce review crawler
Targets: Tiki, Shopee, Lazada
Output: data/raw/{platform}.csv + data/raw/combined.csv
"""

import argparse
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

from crawlers import tiki_crawler, shopee_crawler, lazada_crawler
from crawlers import lazada_recovery
from crawlers.utils import merge_all


def main():
    parser = argparse.ArgumentParser(description="Vietnamese e-commerce review crawler")
    parser.add_argument(
        "--platforms", nargs="+",
        choices=["tiki", "shopee", "lazada", "all"],
        default=["all"],
        help="Which platforms to crawl (default: all)",
    )
    parser.add_argument(
        "--target", type=int, default=800,
        help="Target number of reviews per platform (default: 800)",
    )
    parser.add_argument(
        "--outdir", default="data/raw",
        help="Output directory (default: data/raw)",
    )
    args = parser.parse_args()

    platforms = args.platforms
    if "all" in platforms:
        platforms = ["tiki", "shopee", "lazada"]

    print(f"=== Starting crawl: {platforms} | target={args.target} per platform ===\n")

    if "tiki" in platforms:
        print("--- Tiki ---")
        tiki_crawler.run(target=args.target)
        print()

    if "shopee" in platforms:
        print("--- Shopee ---")
        shopee_crawler.run(target=args.target)
        print()

    if "lazada" in platforms:
        print("--- Lazada ---")
        results = lazada_crawler.run(target=args.target)
        if len(results) < 50:
            print(f"[lazada] Only {len(results)} reviews from main crawler, trying recovery...")
            lazada_recovery.run(target=args.target)
        print()

    print("--- Merging all platforms ---")
    merge_all(out_dir=args.outdir)

    print("--- Supplementing with UIT-VSFC ---")
    from crawlers.uit_vsfc_loader import supplement_data
    supplement_data(os.path.join(args.outdir, "combined.csv"), target_per_class=400)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
