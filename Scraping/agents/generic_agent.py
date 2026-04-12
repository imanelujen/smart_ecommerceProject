"""
agents/generic_agent.py
-----------------------
Generic HTML scraper for any eCommerce site not handled by
the specialized agents. Uses pattern-based heuristics to detect
product cards regardless of theme.
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from .base_agent import BaseAgent, Product


# Common CSS selectors seen across popular eCommerce themes
PRODUCT_CARD_SELECTORS = [
    ".product-card", ".product-item", ".product-tile",
    "[class*='product-grid']", "li.product", ".card--product",
    "[data-product-id]", "[itemtype*='Product']",
]

TITLE_SELECTORS = [
    "h1[itemprop='name']", "h2[itemprop='name']",
    ".product-card__title", ".product__title", ".product-title",
    "h2 a", "h3 a", ".card__heading",
]

PRICE_SELECTORS = [
    "[itemprop='price']", "[class*='price__current']",
    ".product-price", ".price", "[class*='offer-price']",
    ".sale-price", "span.amount",
]


class GenericHTMLAgent(BaseAgent):
    """
    A2A fallback agent for any eCommerce site.

    Detection:  tries to find product cards in the HTML.
    Strategy:   iterative CSS selector attempts on product listing pages.
    """

    name = "generic"

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ------------------------------------------------------------------
    # Detection — always True: this is the last-resort agent
    # ------------------------------------------------------------------

    def detect(self, url: str) -> bool:
        return True   # fallback — always accepts

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def scrape(self, url: str) -> List[dict]:
        raw = []
        page = 1
        while True:
            page_url = self._next_page_url(url, page)
            try:
                self._rate_limit()
                resp = self.session.get(page_url, timeout=15)
                if resp.status_code == 404:
                    break
                soup = BeautifulSoup(resp.text, "lxml")

                cards = self._find_product_cards(soup)
                if not cards:
                    if page == 1:
                        # Try scraping as a single product page
                        single = self._scrape_single_product(soup, url)
                        if single:
                            raw.append(single)
                    break

                for card in cards:
                    item = self._extract_card(card, resp.url)
                    if item:
                        raw.append(item)

                # Check for next page
                if not self._has_next_page(soup):
                    break
                page += 1

            except Exception as e:
                self.logger.error(f"Scrape error page {page}: {e}")
                break

        return raw

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self, raw_products: List[dict]) -> List[Product]:
        products = []
        for item in raw_products:
            try:
                p = Product(
                    title=item.get("title", "").strip(),
                    price=self._safe_float(item.get("price", 0)),
                    currency=item.get("currency", "USD"),
                    url=item.get("url", ""),
                    platform=self.name,
                    availability=item.get("availability", True),
                    rating=item.get("rating"),
                    description=item.get("description", ""),
                    category=item.get("category", ""),
                    image_url=item.get("image_url", ""),
                    raw=item,
                )
                if p.title:
                    products.append(p)
            except Exception as e:
                self.logger.warning(f"Clean error: {e}")
        return products

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_product_cards(self, soup: BeautifulSoup):
        """Try known selectors and return the first non-empty match."""
        for selector in PRODUCT_CARD_SELECTORS:
            cards = soup.select(selector)
            if len(cards) >= 2:   # at least 2 = likely a listing, not a nav
                return cards
        return []

    def _extract_card(self, card, base_url: str) -> Optional[dict]:
        """Pull title, price, URL, image from a product card."""
        title = self._find_text(card, TITLE_SELECTORS)
        if not title:
            return None

        price_str = self._find_text(card, PRICE_SELECTORS)
        link      = card.select_one("a[href]")
        image     = card.select_one("img")

        # Resolve relative URL
        href = link["href"] if link else ""
        if href and not href.startswith("http"):
            match = re.match(r"(https?://[^/]+)", base_url)
            href  = (match.group(1) if match else "") + href

        return {
            "title":     title,
            "price":     price_str or "0",
            "url":       href,
            "image_url": image.get("src", "") if image else "",
        }

    def _scrape_single_product(self, soup: BeautifulSoup, url: str) -> Optional[dict]:
        """Try to extract data when the URL is a single product page."""
        # Look for schema.org Product markup
        title_el = soup.select_one("[itemprop='name']")
        price_el = soup.select_one("[itemprop='price']")
        desc_el  = soup.select_one("[itemprop='description']")
        img_el   = soup.select_one("[itemprop='image']")

        if title_el:
            return {
                "title":       title_el.get_text(strip=True),
                "price":       price_el.get("content") if price_el else "0",
                "url":         url,
                "description": desc_el.get_text(strip=True)[:500] if desc_el else "",
                "image_url":   img_el.get("src", "") if img_el else "",
            }
        return None

    @staticmethod
    def _find_text(element, selectors: List[str]) -> str:
        for sel in selectors:
            found = element.select_one(sel)
            if found:
                return found.get_text(strip=True)
        return ""

    @staticmethod
    def _next_page_url(base_url: str, page: int) -> str:
        if page == 1:
            return base_url
        # Try appending ?page=N or /page/N
        if "?" in base_url:
            return f"{base_url}&page={page}"
        return f"{base_url}?page={page}"

    @staticmethod
    def _has_next_page(soup: BeautifulSoup) -> bool:
        """Check for a 'next' pagination link."""
        return bool(
            soup.select_one("a[rel='next']")
            or soup.select_one(".pagination__next")
            or soup.select_one("a.next")
            or soup.select_one("[aria-label='Next page']")
        )