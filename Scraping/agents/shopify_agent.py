"""
agents/shopify_agent.py
-----------------------
Shopify agent — now captures all 20 required fields.

New vs v1:
  - subcategory, tags from product_type + tags[]
  - price_old (compare_at_price), price_promo, discount_pct
  - stock_quantity (inventory_quantity from variants)
  - variant_count, colors, sizes (from options[])
  - shop_name (from URL), published_at
  - related_products (from metafields if available)
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from .base_agent import BaseAgent, Product


class ShopifyAgent(BaseAgent):
    name = "shopify"
    PRODUCTS_PER_PAGE = 250

    def __init__(self, storefront_token: Optional[str] = None, shop_country: str = ""):
        super().__init__()
        self.storefront_token = storefront_token or os.getenv("SHOPIFY_STOREFRONT_TOKEN")
        self.shop_country = shop_country or os.getenv("SHOP_COUNTRY", "")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
            ),
            "Accept": "application/json, text/html",
        })

    # ── Detection ─────────────────────────────────────────────────────

    def detect(self, url: str) -> bool:
        if "myshopify.com" in url:
            return True
        try:
            resp = self.session.get(url, timeout=10)
            if "Shopify.theme" in resp.text or "shopify-section" in resp.text:
                return True
        except Exception:
            pass
        return False

    # ── Scraping ──────────────────────────────────────────────────────

    def scrape(self, url: str) -> List[dict]:
        base_url = self._extract_base_url(url)
        raw = self._scrape_json_api(base_url)
        if not raw:
            self.logger.warning("JSON API empty, falling back to HTML")
            raw = self._scrape_html(base_url)
        return raw

    def _scrape_json_api(self, base_url: str) -> List[dict]:
        all_products = []
        page = 1
        while True:
            endpoint = f"{base_url}/products.json?limit={self.PRODUCTS_PER_PAGE}&page={page}"
            try:
                self._rate_limit()
                resp = self.session.get(endpoint, timeout=15)
                resp.raise_for_status()
                products = resp.json().get("products", [])
                if not products:
                    break
                # Attach base_url for shop_name extraction
                for p in products:
                    p["_base_url"] = base_url
                all_products.extend(products)
                self.logger.info(f"Page {page}: {len(products)} products")
                if len(products) < self.PRODUCTS_PER_PAGE:
                    break
                page += 1
            except Exception as e:
                self.logger.error(f"Page {page} error: {e}")
                break
        return all_products

    def _scrape_html(self, base_url: str) -> List[dict]:
        raw = []
        page = 1
        while True:
            try:
                self._rate_limit()
                resp = self.session.get(f"{base_url}/collections/all?page={page}", timeout=15)
                if resp.status_code == 404:
                    break
                soup = BeautifulSoup(resp.text, "lxml")
                cards = (
                    soup.select(".product-card") or soup.select(".grid__item")
                    or soup.select("[class*='product']")
                )
                if not cards:
                    break
                for card in cards:
                    title_el = card.select_one("h2, h3, .product-card__title")
                    price_el = card.select_one(".price, [class*='price']")
                    link_el  = card.select_one("a[href*='/products/']")
                    img_el   = card.select_one("img")
                    if not title_el:
                        continue
                    raw.append({
                        "title":     title_el.get_text(strip=True),
                        "price":     price_el.get_text(strip=True) if price_el else "0",
                        "url":       base_url + link_el["href"] if link_el else "",
                        "image_url": img_el.get("src", "") if img_el else "",
                        "_source":   "html",
                        "_base_url": base_url,
                    })
                page += 1
            except Exception as e:
                self.logger.error(f"HTML page {page}: {e}")
                break
        return raw

    # ── Cleaning ──────────────────────────────────────────────────────

    def clean(self, raw_products: List[dict]) -> List[Product]:
        products = []
        for item in raw_products:
            try:
                p = (self._clean_html_item(item)
                     if item.get("_source") == "html"
                     else self._clean_api_item(item))
                if p and p.title:
                    products.append(p)
            except Exception as e:
                self.logger.warning(f"Clean error '{item.get('title','?')}': {e}")
        return products

    def _clean_api_item(self, item: dict) -> Optional[Product]:
        variants = item.get("variants", [{}])
        first    = variants[0] if variants else {}

        # ── Prices ────────────────────────────────────────────────────
        price     = self._safe_float(first.get("price", 0))
        price_old = self._safe_float(first.get("compare_at_price") or 0) or None
        price_promo = price if (price_old and price < price_old) else None
        if price_promo:
            price = price_old   # price_old = original, price_promo = sale price
        discount_pct = self._discount(price_old or price, price_promo or price)

        # ── Stock ─────────────────────────────────────────────────────
        available   = any(v.get("available", False) for v in variants)
        stock_qty   = sum(
            self._safe_int(v.get("inventory_quantity", 0))
            for v in variants
            if self._safe_int(v.get("inventory_quantity", 0)) > 0
        ) or None

        # ── Variants ──────────────────────────────────────────────────
        options     = item.get("options", [])
        colors      = self._extract_option(options, ["color", "colour", "couleur"])
        sizes       = self._extract_option(options, ["size", "taille"])

        # ── Tags ──────────────────────────────────────────────────────
        tags = ", ".join(item.get("tags", []) if isinstance(item.get("tags"), list)
                         else str(item.get("tags", "")).split(","))

        # ── Shop info ─────────────────────────────────────────────────
        base_url  = item.get("_base_url", "")
        shop_name = re.sub(r"https?://", "", base_url).split(".")[0]

        # ── Image ─────────────────────────────────────────────────────
        images    = item.get("images", [])
        image_url = images[0].get("src", "") if images else ""

        # ── Category / subcategory ────────────────────────────────────
        product_type = item.get("product_type", "")
        parts = product_type.split("/", 1)
        category    = parts[0].strip()
        subcategory = parts[1].strip() if len(parts) > 1 else ""

        return Product(
            product_id    = str(item.get("id", "")),
            title         = item.get("title", "").strip(),
            description   = self._strip_html(item.get("body_html", ""))[:600],
            category      = category,
            subcategory   = subcategory,
            brand         = item.get("vendor", ""),
            image_url     = image_url,
            tags          = tags[:200],
            url           = f"{base_url}/products/{item.get('handle', '')}",
            platform      = self.name,
            price         = price,
            price_promo   = price_promo,
            price_old     = price_old,
            discount_pct  = discount_pct,
            currency      = "USD",
            availability  = available,
            stock_quantity= stock_qty,
            variant_count = len(variants),
            colors        = colors,
            sizes         = sizes,
            shop_name     = shop_name,
            shop_country  = self.shop_country,
            published_at  = str(item.get("published_at", "")),
            raw           = item,
        )

    def _clean_html_item(self, item: dict) -> Optional[Product]:
        base_url  = item.get("_base_url", "")
        shop_name = re.sub(r"https?://", "", base_url).split(".")[0]
        return Product(
            title        = item.get("title", "").strip(),
            price        = self._safe_float(item.get("price", 0)),
            currency     = "USD",
            url          = item.get("url", ""),
            platform     = self.name,
            image_url    = item.get("image_url", ""),
            shop_name    = shop_name,
            shop_country = self.shop_country,
            raw          = item,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_option(options: list, keywords: List[str]) -> str:
        for opt in options:
            name = opt.get("name", "").lower()
            if any(k in name for k in keywords):
                return ", ".join(opt.get("values", []))
        return ""