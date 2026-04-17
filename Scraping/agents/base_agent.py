"""
agents/base_agent.py
--------------------
Abstract base class for all A2A scraping agents.
Product dataclass covers all 9 data groups required by the project spec:
  1. Descriptives      4. Stock/dispo    7. Marketing
  2. Prix              5. Variantes      8. Temporelles
  3. Popularité        6. Vendeur        9. Textuelles
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import time
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)


@dataclass
class Product:
    """
    Normalized product model — 20 variables aligned with project spec.
    Maps directly to the final dataset columns used by ML algorithms.
    """

    # ── 1. Descriptives ───────────────────────────────────────────────
    product_id: str = ""
    title: str = ""
    description: str = ""
    category: str = ""
    subcategory: str = ""
    brand: str = ""
    image_url: str = ""
    tags: str = ""            # comma-separated keywords
    url: str = ""
    platform: str = ""        # "shopify" | "woocommerce" | "generic"

    # ── 2. Prix ───────────────────────────────────────────────────────
    price: float = 0.0
    price_promo: Optional[float] = None   # prix promotionnel
    price_old: Optional[float] = None     # ancien prix (compare_at)
    discount_pct: Optional[float] = None  # remise calculée en %
    currency: str = "USD"

    # ── 3. Popularité ─────────────────────────────────────────────────
    rating: Optional[float] = None
    review_count: int = 0
    category_rank: Optional[int] = None

    # ── 4. Stock & disponibilité ──────────────────────────────────────
    availability: bool = True
    stock_quantity: Optional[int] = None
    delivery_days: Optional[int] = None

    # ── 5. Variantes ──────────────────────────────────────────────────
    variant_count: int = 1
    colors: str = ""
    sizes: str = ""

    # ── 6. Vendeur / shop ─────────────────────────────────────────────
    shop_name: str = ""
    shop_country: str = ""
    shop_product_count: Optional[int] = None
    vendor: str = ""  # Ajouté pour compatibilité avec les tests

    # ── 7. Marketing / associations ───────────────────────────────────
    related_products: str = ""

    # ── 8. Temporelles ────────────────────────────────────────────────
    published_at: str = ""
    scraped_at: str = ""

    # ── 9. Textuelles (NLP) ───────────────────────────────────────────
    customer_reviews: str = ""

    # Internal
    raw: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "raw"}
        if d["discount_pct"] is None and d.get("price_old") and d["price_old"] > 0:
            d["discount_pct"] = round(
                (d["price_old"] - d["price"]) / d["price_old"] * 100, 1
            )
        return d

    @property
    def is_on_sale(self) -> bool:
        return self.price_promo is not None and self.price_promo < self.price


class BaseAgent(ABC):
    name: str = "base"
    MIN_DELAY: float = 0.5
    MAX_DELAY: float = 2.0

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = None

    @abstractmethod
    def detect(self, url: str) -> bool: ...

    @abstractmethod
    def scrape(self, url: str) -> List[dict]: ...

    @abstractmethod
    def clean(self, raw_products: List[dict]) -> List[Product]: ...

    def run(self, url: str) -> List[Product]:
        self.logger.info(f"Starting scrape: {url}")
        raw = self.scrape(url)
        self.logger.info(f"Fetched {len(raw)} raw products")
        products = self.clean(raw)
        self.logger.info(f"Cleaned to {len(products)} products")
        return products

    def _rate_limit(self):
        time.sleep(random.uniform(self.MIN_DELAY, self.MAX_DELAY))

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        s = "".join(c for c in str(value).strip() if c.isdigit() or c in ".,")
        if not s:
            return default
        if "," in s and "." in s:
            if s.rfind(".") > s.rfind(","):
                s = s.replace(",", "")
            else:
                s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            parts = s.split(",")
            if len(parts) == 2 and len(parts[1]) == 2:
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return default

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(str(value).replace(",", "").strip())
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _discount(original: float, current: float) -> Optional[float]:
        if original and original > current > 0:
            return round((original - current) / original * 100, 1)
        return None

    @staticmethod
    def _strip_html(html: str) -> str:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html or "", "lxml").get_text(separator=" ", strip=True)

    @staticmethod
    def _extract_base_url(url: str) -> str:
        import re
        m = re.match(r"(https?://[^/]+)", url)
        return m.group(1) if m else url