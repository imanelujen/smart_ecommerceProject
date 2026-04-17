import pytest
from unittest.mock import MagicMock, patch
from Scraping.agents.base_agent import Product, BaseAgent
from Scraping.agents.shopify_agent import ShopifyAgent
from Scraping.agents.woocommerce_agent import WooCommerceAgent
from Scraping.agents.generic_agent import GenericHTMLAgent
from Scraping.orchestrator import Orchestrator


# ======================================================================
# BaseAgent helpers
# ======================================================================

class TestBaseHelpers:
    def setup_method(self):
        self.agent = ShopifyAgent()   # any concrete subclass

    def test_safe_float_dollar(self):
        assert self.agent._safe_float("$19.99") == 19.99

    def test_safe_float_comma_decimal(self):
        assert self.agent._safe_float("1,299.00") == 1299.0

    def test_safe_float_euro(self):
        assert self.agent._safe_float("€9,99") == 9.99

    def test_safe_float_empty(self):
        assert self.agent._safe_float("") == 0.0

    def test_safe_float_none(self):
        assert self.agent._safe_float(None) == 0.0

    def test_safe_int_comma(self):
        assert self.agent._safe_int("1,234") == 1234

    def test_safe_int_invalid(self):
        assert self.agent._safe_int("N/A") == 0


# ======================================================================
# ShopifyAgent
# ======================================================================

class TestShopifyAgent:
    def setup_method(self):
        self.agent = ShopifyAgent()

    def test_detect_myshopify_url(self):
        assert self.agent.detect("https://demo-store.myshopify.com")

    def test_extract_base_url(self):
        url = "https://demo-store.myshopify.com/collections/all"
        assert ShopifyAgent._extract_base_url(url) == "https://demo-store.myshopify.com"

    def test_clean_api_item(self):
        raw = {
            "title": "Test T-Shirt",
            "handle": "test-t-shirt",
            "vendor": "ACME",
            "product_type": "Clothing",
            "body_html": "<p>Great quality shirt.</p>",
            "variants": [{"price": "29.99", "available": True}],
            "images": [{"src": "https://cdn.example.com/shirt.jpg"}],
        }
        product = self.agent._clean_api_item(raw)
        assert product.title == "Test T-Shirt"
        assert product.price == 29.99
        assert product.availability is True
        assert product.platform == "shopify"
        assert product.vendor == "ACME"
        assert "Great quality shirt" in product.description

    def test_clean_skips_empty_title(self):
        raw = {
            "title": "",
            "variants": [{"price": "9.99", "available": True}],
        }
        product = self.agent._clean_api_item(raw)
        # title is empty — product should still be created but with empty title
        assert product.title == ""

    @patch("agents.shopify_agent.requests.Session.get")
    def test_scrape_json_api_paginated(self, mock_get):
        """Verifies pagination stops when a partial page is received."""
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = {
            "products": [{"title": f"Product {i}", "variants": [], "images": []} for i in range(250)]
        }
        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = {
            "products": [{"title": "Product last", "variants": [], "images": []}]
        }
        mock_get.side_effect = [page1, page2]

        agent = ShopifyAgent()
        agent.MIN_DELAY = 0
        results = agent._scrape_json_api("https://example.myshopify.com")
        assert len(results) == 251


# ======================================================================
# WooCommerceAgent
# ======================================================================

class TestWooCommerceAgent:
    def setup_method(self):
        self.agent = WooCommerceAgent(consumer_key="ck_test", consumer_secret="cs_test")

    def test_clean_item_basic(self):
        raw = {
            "name": "Blue Hoodie",
            "price": "45.00",
            "regular_price": "45.00",
            "stock_status": "instock",
            "average_rating": "4.5",
            "rating_count": 23,
            "categories": [{"name": "Apparel"}],
            "images": [{"src": "https://example.com/hoodie.jpg"}],
            "description": "<p>Warm and cozy hoodie.</p>",
            "permalink": "https://example.com/products/blue-hoodie",
        }
        product = self.agent._clean_item(raw)
        assert product.title == "Blue Hoodie"
        assert product.price == 45.0
        assert product.availability is True
        assert product.rating == 4.5
        assert product.review_count == 23
        assert product.category == "Apparel"
        assert product.platform == "woocommerce"

    def test_clean_out_of_stock(self):
        raw = {
            "name": "Rare Item",
            "price": "99.00",
            "stock_status": "outofstock",
            "categories": [],
            "images": [],
            "description": "",
        }
        product = self.agent._clean_item(raw)
        assert product.availability is False


# ======================================================================
# GenericHTMLAgent
# ======================================================================

class TestGenericAgent:
    def setup_method(self):
        self.agent = GenericHTMLAgent()

    def test_detect_always_true(self):
        assert self.agent.detect("https://any-random-site.com")

    def test_next_page_url(self):
        assert GenericHTMLAgent._next_page_url("https://site.com/shop", 1) == "https://site.com/shop"
        assert GenericHTMLAgent._next_page_url("https://site.com/shop", 2) == "https://site.com/shop?page=2"

    def test_clean_valid(self):
        raw = [
            {"title": "Widget X", "price": "$12.00", "url": "https://example.com/widget-x"},
            {"title": "", "price": "$5.00", "url": ""},   # empty title — should be excluded
        ]
        products = self.agent.clean(raw)
        assert len(products) == 1
        assert products[0].title == "Widget X"
        assert products[0].price == 12.0


# ======================================================================
# Orchestrator
# ======================================================================

class TestOrchestrator:
    def test_detect_agent_shopify(self, tmp_path):
        orch = Orchestrator(output_dir=str(tmp_path))
        with patch.object(orch.agents[0], "detect", return_value=True):
            agent = orch._detect_agent("https://test.myshopify.com")
        assert agent.name == "shopify"

    def test_deduplicate(self, tmp_path):
        import pandas as pd
        orch = Orchestrator(output_dir=str(tmp_path))
        df = pd.DataFrame([
            {"title": "A", "price": 10.0, "platform": "shopify"},
            {"title": "A", "price": 10.0, "platform": "shopify"},   # duplicate
            {"title": "B", "price": 20.0, "platform": "woocommerce"},
        ])
        result = orch._deduplicate(df)
        assert len(result) == 2

    def test_add_metadata_price_band(self, tmp_path):
        import pandas as pd
        orch = Orchestrator(output_dir=str(tmp_path))
        df = pd.DataFrame([
            {"title": "Cheap", "price": 5.0,  "platform": "shopify"},
            {"title": "Mid",   "price": 50.0, "platform": "shopify"},
        ])
        df = orch._add_metadata(df)
        assert "price_band" in df.columns
        assert "scraped_at" in df.columns
        assert df["price_band"].iloc[0] == "<$10"
        assert df["price_band"].iloc[1] == "$30-100"