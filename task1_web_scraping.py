"""
============================================================
CODEALPHA INTERNSHIP — TASK 1: WEB SCRAPING
============================================================
Description : Scrapes book data from books.toscrape.com
              (a legal, public practice site) and saves
              the results to a CSV file for further analysis.
Libraries   : requests, BeautifulSoup, pandas
Install     : pip install requests beautifulsoup4 pandas
============================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# ── CONFIG ──────────────────────────────────────────────────
BASE_URL  = "https://books.toscrape.com/catalogue/"
START_URL = "https://books.toscrape.com/catalogue/page-1.html"
MAX_PAGES = 5          # scrape first 5 pages (~100 books)
OUTPUT    = "books_data.csv"

# ── HELPERS ─────────────────────────────────────────────────
STAR_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def get_star_rating(tag) -> int:
    """Convert word-based star class to an integer (1-5)."""
    classes = tag.find("p", class_="star-rating")["class"]
    word = classes[1]          # e.g. "Three"
    return STAR_MAP.get(word, 0)

def clean_price(raw: str) -> float:
    """Strip currency symbol and return float."""
    return float(raw.replace("Â£", "").replace("£", "").strip())

# ── SCRAPER ─────────────────────────────────────────────────
def scrape_books(start_url: str, max_pages: int) -> pd.DataFrame:
    records = []
    url     = start_url

    for page_num in range(1, max_pages + 1):
        print(f"  Scraping page {page_num}: {url}")
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"  ✗ Failed (status {response.status_code}). Stopping.")
            break

        soup  = BeautifulSoup(response.text, "html.parser")
        books = soup.find_all("article", class_="product_pod")

        for book in books:
            title  = book.h3.a["title"]
            price  = clean_price(book.find("p", class_="price_color").text)
            rating = get_star_rating(book)
            avail  = book.find("p", class_="instock availability").text.strip()
            link   = BASE_URL + book.h3.a["href"].replace("../", "")

            records.append({
                "Title"        : title,
                "Price (£)"    : price,
                "Star Rating"  : rating,
                "Availability" : avail,
                "URL"          : link,
            })

        # Find next-page link
        next_btn = soup.find("li", class_="next")
        if not next_btn:
            print("  ✓ No more pages — scraping complete.")
            break
        url = BASE_URL + next_btn.a["href"]
        time.sleep(1)   # be polite: 1-second delay between pages

    return pd.DataFrame(records)

# ── ANALYSIS ────────────────────────────────────────────────
def analyse(df: pd.DataFrame) -> None:
    print("\n========== BASIC STATISTICS ==========")
    print(f"Total books scraped : {len(df)}")
    print(f"Average price       : £{df['Price (£)'].mean():.2f}")
    print(f"Most expensive      : £{df['Price (£)'].max():.2f}")
    print(f"Cheapest            : £{df['Price (£)'].min():.2f}")
    print(f"\nRating distribution:\n{df['Star Rating'].value_counts().sort_index()}")
    print(f"\nTop 5 most expensive books:\n{df.nlargest(5, 'Price (£)')[['Title','Price (£)','Star Rating']].to_string(index=False)}")
    print(f"\nTop 5 highest-rated (tie-broken by price desc):")
    top = df.sort_values(["Star Rating", "Price (£)"], ascending=[False, False]).head(5)
    print(top[["Title", "Star Rating", "Price (£)"]].to_string(index=False))

# ── MAIN ────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  CODEALPHA — Task 1: Web Scraping")
    print("=" * 55)

    df = scrape_books(START_URL, MAX_PAGES)

    if df.empty:
        print("No data collected. Check your internet connection.")
        return

    df.to_csv(OUTPUT, index=False)
    print(f"\n✓ Data saved to '{OUTPUT}'")

    analyse(df)
    print("\n✓ Task 1 complete!")

if __name__ == "__main__":
    main()
