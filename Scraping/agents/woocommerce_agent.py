"""
agents/woocommerce_agent.py
---------------------------
WooCommerce agent — now captures all 20 required fields.

New vs v1:
  - subcategory (from categories[1] if present)
  - tags (from WC tags endpoint)
  - price_old (regular_price), price_promo (sale_price), discount_pct
  - stock_quantity (stock_quantity field)
  - variant_count, colors, sizes (from attributes[])
  - shop_name, shop_country, shop_product_count
  - published_at (date_created)
  - customer_reviews (from /reviews endpoint, sample of 3)
"""

import os
import re
import requests
from typing import List, Optional
from .base_agent import BaseAgent, Product


class WooCommerceAgent(BaseAgent):
    name = "woocommerce"
    PER_PAGE = 100

    def __init__(
        self,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
        shop_name: str = "",
        shop_country: str = "",
    ):
        super().__init__()
        self.consumer_key    = consumer_key    or os.getenv("WC_CONSUMER_KEY", "")
        self.consumer_secret = consumer_secret or os.getenv("WC_CONSUMER_SECRET", "")
        self.shop_name_override    = shop_name    or os.getenv("SHOP_NAME", "")
        self.shop_country_override = shop_country or os.getenv("SHOP_COUNTRY", "")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SmartEcommerceBot/2.0"})
        self._shop_info_cache: dict = {}   # cached from /wp-json/wc/v3/system_status

    # ── Detection ─────────────────────────────────────────────────────

    def detect(self, url: str) -> bool:
        base = self._extract_base_url(url)
        try:
            resp = self.session.get(f"{base}/wp-json/wc/v3/", timeout=8)
            if resp.status_code in (200, 401):
                return True
        except Exception:
            pass
        try:
            resp = self.session.get(base, timeout=8)
            if "woocommerce" in resp.text.lower():
                return True
        except Exception:
            pass
        return False

    # ── Scraping ──────────────────────────────────────────────────────

    def scrape(self, url: str) -> List[dict]:
        base = self._extract_base_url(url)
        if self.consumer_key and self.consumer_secret:
            self._fetch_shop_info(base)
            raw = self._scrape_rest_api(base)
            self._enrich_with_reviews(base, raw)
        else:
            self.logger.warning("No WC credentials — using Selenium fallback")
            raw = self._scrape_selenium(base)
        return raw

    def _scrape_rest_api(self, base_url: str) -> List[dict]:
        all_products = []
        page = 1
        auth = (self.consumer_key, self.consumer_secret)
        while True:
            endpoint = (
                f"{base_url}/wp-json/wc/v3/products"
                f"?per_page={self.PER_PAGE}&page={page}&status=publish"
            )
            try:
                self._rate_limit()
                resp = self.session.get(endpoint, auth=auth, timeout=20)
                resp.raise_for_status()
                products = resp.json()
                if not products:
                    break
                for p in products:
                    p["_base_url"] = base_url
                all_products.extend(products)
                self.logger.info(f"Page {page}: {len(products)} products")
                total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
                if page >= total_pages:
                    break
                page += 1
            except Exception as e:
                self.logger.error(f"API page {page}: {e}")
                break
        return all_products

    def _fetch_shop_info(self, base_url: str):
        """Fetch shop-level metadata: product count."""
        try:
            auth = (self.consumer_key, self.consumer_secret)
            resp = self.session.get(
                f"{base_url}/wp-json/wc/v3/reports/products/totals",
                auth=auth, timeout=10
            )
            if resp.ok:
                data = resp.json()
                total = sum(item.get("total", 0) for item in data)
                self._shop_info_cache["product_count"] = total
        except Exception:
            pass

    def _enrich_with_reviews(self, base_url: str, products: List[dict]):
        """
        Fetch up to 3 review texts per product and attach as _reviews.
        Uses /wp-json/wc/v3/products/{id}/reviews
        """
        auth = (self.consumer_key, self.consumer_secret)
        for product in products[:50]:   # limit to avoid excessive requests
            pid = product.get("id")
            if not pid:
                continue
            try:
                self._rate_limit()
                resp = self.session.get(
                    f"{base_url}/wp-json/wc/v3/products/{pid}/reviews?per_page=3",
                    auth=auth, timeout=10
                )
                if resp.ok:
                    reviews = resp.json()
                    texts = [r.get("review", "") for r in reviews if r.get("review")]
                    product["_reviews"] = " | ".join(
                        self._strip_html(t)[:150] for t in texts
                    )
            except Exception:
                pass

    def _scrape_selenium(self, base_url: str) -> List[dict]:
        raw = []
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from bs4 import BeautifulSoup

            opts = Options()
            opts.add_argument("--headless")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            driver = webdriver.Chrome(options=opts)
            wait   = WebDriverWait(driver, 10)
            page   = 1

            while True:
                driver.get(f"{base_url}/shop/page/{page}/")
                self._rate_limit()
                try:
                    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "products")))
                except Exception:
                    break
                soup  = BeautifulSoup(driver.page_source, "lxml")
                cards = soup.select("li.product")
                if not cards:
                    break
                for card in cards:
                    title_el = card.select_one(".woocommerce-loop-product__title, h2")
                    price_el = card.select_one(".price .woocommerce-Price-amount")
                    link_el  = card.select_one("a.woocommerce-LoopProduct-link, a")
                    img_el   = card.select_one("img")
                    raw.append({
                        "name":      title_el.get_text(strip=True) if title_el else "",
                        "price":     price_el.get_text(strip=True) if price_el else "0",
                        "permalink": link_el["href"] if link_el else "",
                        "images":    [{"src": img_el["src"]}] if img_el else [],
                        "_source":   "selenium",
                        "_base_url": base_url,
                    })
                if not soup.select_one("a.next.page-numbers"):
                    break
                page += 1
            driver.quit()
        except Exception as e:
            self.logger.error(f"Selenium error: {e}")
        return raw

    # ── Cleaning ──────────────────────────────────────────────────────

    def clean(self, raw_products: List[dict]) -> List[Product]:
        products = []
        for item in raw_products:
            try:
                p = self._clean_item(item)
                if p and p.title:
                    products.append(p)
            except Exception as e:
                self.logger.warning(f"Clean error '{item.get('name','?')}': {e}")
        return products

    def _clean_item(self, item: dict) -> Optional[Product]:
        # ── Prices ────────────────────────────────────────────────────
        regular = self._safe_float(item.get("regular_price") or item.get("price") or 0)
        sale    = self._safe_float(item.get("sale_price") or 0) or None
        price   = regular
        price_promo = sale if (sale and sale < regular) else None
        discount_pct = self._discount(regular, price_promo) if price_promo else None

        # ── Stock ─────────────────────────────────────────────────────
        stock_status = item.get("stock_status", "instock")
        available    = stock_status == "instock"
        stock_qty    = self._safe_int(item.get("stock_quantity")) if item.get("stock_quantity") else None

        # ── Variants / attributes ─────────────────────────────────────
        attributes   = item.get("attributes", [])
        colors       = self._extract_attr(attributes, ["color", "colour", "couleur"])
        sizes        = self._extract_attr(attributes, ["size", "taille"])
        variant_count = len(item.get("variations", [])) or 1

        # ── Categories ────────────────────────────────────────────────
        cats        = item.get("categories", [])
        category    = cats[0]["name"] if cats else ""
        subcategory = cats[1]["name"] if len(cats) > 1 else ""

        # ── Tags ──────────────────────────────────────────────────────
        tags_list = item.get("tags", [])
        tags = ", ".join(t["name"] for t in tags_list if t.get("name"))

        # ── Rating ────────────────────────────────────────────────────
        rating       = self._safe_float(item.get("average_rating") or 0) or None
        review_count = self._safe_int(item.get("rating_count", 0))

        # ── Image ─────────────────────────────────────────────────────
        images    = item.get("images", [])
        image_url = images[0].get("src", "") if images else ""

        # ── Description ───────────────────────────────────────────────
        raw_desc = item.get("description") or item.get("short_description", "")
        description = self._strip_html(raw_desc)[:600]

        # ── Shop info ─────────────────────────────────────────────────
        base_url  = item.get("_base_url", "")
        shop_name = (
            self.shop_name_override
            or re.sub(r"https?://", "", base_url).rstrip("/").split("/")[0]
        )

        return Product(
            product_id        = str(item.get("id", "")),
            title             = item.get("name", "").strip(),
            description       = description,
            category          = category,
            subcategory       = subcategory,
            brand             = item.get("sku", ""),
            image_url         = image_url,
            tags              = tags[:200],
            url               = item.get("permalink", ""),
            platform          = self.name,
            price             = price,
            price_promo       = price_promo,
            price_old         = regular if price_promo else None,
            discount_pct      = discount_pct,
            currency          = "USD",
            rating            = rating,
            review_count      = review_count,
            availability      = available,
            stock_quantity    = stock_qty,
            variant_count     = variant_count,
            colors            = colors,
            sizes             = sizes,
            shop_name         = shop_name,
            shop_country      = self.shop_country_override,
            shop_product_count= self._shop_info_cache.get("product_count"),
            published_at      = str(item.get("date_created", "")),
            customer_reviews  = item.get("_reviews", ""),
            raw               = item,
        )

    @staticmethod
    def _extract_attr(attributes: list, keywords: List[str]) -> str:
        for attr in attributes:
            name = attr.get("name", "").lower()
            if any(k in name for k in keywords):
                options = attr.get("options", [])
                return ", ".join(options)
        return ""